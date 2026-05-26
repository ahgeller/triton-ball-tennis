// Motion-only C++ probe for tennis-ball raw masks.
//
// This is deliberately not a tracker. It reads video frames, compares the
// current adaptive S/V foreground mask with a tighter temporal mask biased
// toward small fast components, then writes an overlay video plus JSON stats.
//
// Build idea on Windows from a VS Developer PowerShell:
//   cl /std:c++17 /O2 /EHsc /I %CONDA_PREFIX%\Library\include 3DtrackingV1\archived_tools\raw_motion_probe.cpp ^
//      /link /LIBPATH:%CONDA_PREFIX%\Library\lib opencv_core490.lib opencv_imgproc490.lib ^
//      opencv_videoio490.lib opencv_imgcodecs490.lib /OUT:3DtrackingV1\archived_tools\raw_motion_probe.exe

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string input = "input_videos/PomonaPitzer Women vs. UCSD-cut-merged-1773082079341.mp4";
    std::string output = "output_videos/pomona_raw_motion_probe_cpp.mp4";
    std::string json = "output_videos/pomona_raw_motion_probe_cpp.json";
    int max_frames = 0;
    int progress_every = 200;

    double motion_thresh = 11.0;
    double motion_k_std = 3.0;
    double motion_v_min = 40.0;
    double alpha = 0.02;
    double motion_alpha = 0.015;

    double temporal_hi = 18.0;
    double temporal_lo = 8.0;
    double temporal_very_hi = 36.0;
    int open_size = 0;
    int close_size = 2;

    int min_area = 2;
    int max_area = 260;
    int max_dim = 38;
    double max_aspect = 5.5;
    double min_fill = 0.14;
};

struct ComponentCounts {
    int components = 0;
    int kept = 0;
    int rejected_large = 0;
    int rejected_shape = 0;
};

struct FrameStats {
    int frame = 0;
    int old_raw_px = 0;
    int new_raw_px = 0;
    int new_components = 0;
    int kept_components = 0;
    int rejected_large = 0;
    int rejected_shape = 0;
    double motion_ms = 0.0;
};

using Clock = std::chrono::steady_clock;

double elapsed_ms(const Clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

double elapsed_sec(const Clock::time_point& start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

bool has_next_value(int i, int argc) {
    return i + 1 < argc;
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto need_value = [&](const std::string& name) -> std::string {
            if (!has_next_value(i, argc)) {
                throw std::runtime_error("Missing value for " + name);
            }
            return argv[++i];
        };

        if (key == "--input") args.input = need_value(key);
        else if (key == "--output") args.output = need_value(key);
        else if (key == "--json") args.json = need_value(key);
        else if (key == "--max-frames") args.max_frames = std::stoi(need_value(key));
        else if (key == "--progress-every") args.progress_every = std::stoi(need_value(key));
        else if (key == "--motion-thresh") args.motion_thresh = std::stod(need_value(key));
        else if (key == "--motion-k-std") args.motion_k_std = std::stod(need_value(key));
        else if (key == "--motion-v-min") args.motion_v_min = std::stod(need_value(key));
        else if (key == "--alpha") args.alpha = std::stod(need_value(key));
        else if (key == "--motion-alpha") args.motion_alpha = std::stod(need_value(key));
        else if (key == "--temporal-hi") args.temporal_hi = std::stod(need_value(key));
        else if (key == "--temporal-lo") args.temporal_lo = std::stod(need_value(key));
        else if (key == "--temporal-very-hi") args.temporal_very_hi = std::stod(need_value(key));
        else if (key == "--open-size") args.open_size = std::stoi(need_value(key));
        else if (key == "--close-size") args.close_size = std::stoi(need_value(key));
        else if (key == "--min-area") args.min_area = std::stoi(need_value(key));
        else if (key == "--max-area") args.max_area = std::stoi(need_value(key));
        else if (key == "--max-dim") args.max_dim = std::stoi(need_value(key));
        else if (key == "--max-aspect") args.max_aspect = std::stod(need_value(key));
        else if (key == "--min-fill") args.min_fill = std::stod(need_value(key));
        else if (key == "--help" || key == "-h") {
            std::cout
                << "raw_motion_probe.cpp options:\n"
                << "  --input PATH --output PATH --json PATH\n"
                << "  --max-frames N --progress-every N\n"
                << "  --motion-thresh F --motion-k-std F --motion-v-min F\n"
                << "  --temporal-hi F --temporal-lo F --temporal-very-hi F\n"
                << "  --min-area N --max-area N --max-dim N --max-aspect F --min-fill F\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + key);
        }
    }
    return args;
}

std::string json_escape(const std::string& s) {
    std::ostringstream out;
    for (char ch : s) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << ch; break;
        }
    }
    return out.str();
}

cv::Mat combined_delta(
    const cv::Mat& prev_hsv,
    const cv::Mat& curr_hsv,
    const cv::Mat& prev_gray,
    const cv::Mat& curr_gray
) {
    std::vector<cv::Mat> prev_ch;
    std::vector<cv::Mat> curr_ch;
    cv::split(prev_hsv, prev_ch);
    cv::split(curr_hsv, curr_ch);

    cv::Mat v_diff;
    cv::Mat s_diff;
    cv::Mat gray_diff;
    cv::absdiff(curr_ch[2], prev_ch[2], v_diff);
    cv::absdiff(curr_ch[1], prev_ch[1], s_diff);
    cv::absdiff(curr_gray, prev_gray, gray_diff);

    cv::Mat s_scaled;
    s_diff.convertTo(s_scaled, CV_32F, 1.25);
    cv::threshold(s_scaled, s_scaled, 255.0, 255.0, cv::THRESH_TRUNC);
    s_scaled.convertTo(s_scaled, CV_8U);

    cv::Mat tmp;
    cv::Mat combo;
    cv::max(v_diff, s_scaled, tmp);
    cv::max(gray_diff, tmp, combo);
    cv::GaussianBlur(combo, combo, cv::Size(3, 3), 0.0);
    return combo;
}

cv::Mat adaptive_sv_raw(
    const cv::Mat& curr_hsv,
    const cv::Mat& bg_v,
    const cv::Mat& bg_s,
    const cv::Mat& var_v,
    const cv::Mat& var_s,
    const Args& args
) {
    std::vector<cv::Mat> ch;
    cv::split(curr_hsv, ch);
    cv::Mat v;
    cv::Mat s;
    ch[2].convertTo(v, CV_32F);
    ch[1].convertTo(s, CV_32F);

    cv::Mat dv = v - bg_v;
    cv::Mat ds = s - bg_s;
    cv::multiply(dv, dv, dv);
    cv::multiply(ds, ds, ds);

    const double thr2 = args.motion_thresh * args.motion_thresh;
    const double k2 = args.motion_k_std * args.motion_k_std;
    cv::Mat v_thr = var_v * k2 + thr2;
    cv::Mat s_thr = (var_s * k2 + thr2) * 1.5;

    cv::Mat mv;
    cv::Mat ms;
    cv::Mat bright;
    cv::compare(dv, v_thr, mv, cv::CMP_GT);
    cv::compare(ds, s_thr, ms, cv::CMP_GT);
    cv::compare(v, args.motion_v_min, bright, cv::CMP_GT);

    cv::Mat raw;
    cv::bitwise_or(mv, ms, raw);
    cv::bitwise_and(raw, bright, raw);
    return raw;
}

void update_background(
    const cv::Mat& curr_hsv,
    cv::Mat& bg_v,
    cv::Mat& bg_s,
    cv::Mat& var_v,
    cv::Mat& var_s,
    const cv::Mat& old_raw,
    const Args& args
) {
    std::vector<cv::Mat> ch;
    cv::split(curr_hsv, ch);
    cv::Mat v;
    cv::Mat s;
    ch[2].convertTo(v, CV_32F);
    ch[1].convertTo(s, CV_32F);

    cv::Mat alpha(bg_v.size(), CV_32F, cv::Scalar(args.alpha));
    alpha.setTo(args.motion_alpha, old_raw);
    cv::Mat inv_alpha = 1.0 - alpha;

    cv::Mat dv = v - bg_v;
    cv::Mat ds = s - bg_s;
    cv::multiply(dv, dv, dv);
    cv::multiply(ds, ds, ds);

    var_v = var_v.mul(inv_alpha) + dv.mul(alpha);
    var_s = var_s.mul(inv_alpha) + ds.mul(alpha);
    bg_v = bg_v.mul(inv_alpha) + v.mul(alpha);
    bg_s = bg_s.mul(inv_alpha) + s.mul(alpha);
}

cv::Mat component_filter(
    const cv::Mat& mask,
    const Args& args,
    ComponentCounts& counts,
    std::vector<cv::Rect>& boxes
) {
    counts = ComponentCounts{};
    boxes.clear();
    if (mask.empty() || cv::countNonZero(mask) == 0) {
        return cv::Mat::zeros(mask.size(), CV_8U);
    }

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int n = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);
    counts.components = std::max(0, n - 1);

    std::vector<std::uint8_t> keep(static_cast<std::size_t>(n), 0);
    boxes.reserve(static_cast<std::size_t>(n));

    for (int i = 1; i < n; ++i) {
        const int area = stats.at<int>(i, cv::CC_STAT_AREA);
        const int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        const int y = stats.at<int>(i, cv::CC_STAT_TOP);
        const int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        if (area < args.min_area) {
            counts.rejected_shape++;
            continue;
        }
        if (area > args.max_area || w > args.max_dim || h > args.max_dim) {
            counts.rejected_large++;
            continue;
        }
        const double aspect = static_cast<double>(std::max(w, h)) / std::max(1, std::min(w, h));
        const double fill = static_cast<double>(area) / std::max(1, w * h);
        if (aspect > args.max_aspect || fill < args.min_fill) {
            counts.rejected_shape++;
            continue;
        }
        keep[static_cast<std::size_t>(i)] = 255;
        boxes.emplace_back(x, y, w, h);
        counts.kept++;
    }

    cv::Mat out(mask.size(), CV_8U, cv::Scalar(0));
    for (int y = 0; y < labels.rows; ++y) {
        const int* src = labels.ptr<int>(y);
        std::uint8_t* dst = out.ptr<std::uint8_t>(y);
        for (int x = 0; x < labels.cols; ++x) {
            dst[x] = keep[static_cast<std::size_t>(src[x])];
        }
    }
    return out;
}

cv::Mat new_raw_motion(
    const cv::Mat& prev_hsv,
    const cv::Mat& curr_hsv,
    const cv::Mat* next_hsv,
    const cv::Mat& prev_gray,
    const cv::Mat& curr_gray,
    const cv::Mat* next_gray,
    const cv::Mat& old_raw,
    const Args& args,
    ComponentCounts& counts,
    std::vector<cv::Rect>& boxes
) {
    cv::Mat d_prev = combined_delta(prev_hsv, curr_hsv, prev_gray, curr_gray);
    cv::Mat temporal;
    cv::compare(d_prev, args.temporal_hi, temporal, cv::CMP_GE);

    if (next_hsv != nullptr && next_gray != nullptr) {
        cv::Mat d_next = combined_delta(curr_hsv, *next_hsv, curr_gray, *next_gray);
        cv::Mat p_hi, p_lo, n_hi, n_lo, p_very;
        cv::compare(d_prev, args.temporal_hi, p_hi, cv::CMP_GE);
        cv::compare(d_prev, args.temporal_lo, p_lo, cv::CMP_GE);
        cv::compare(d_next, args.temporal_hi, n_hi, cv::CMP_GE);
        cv::compare(d_next, args.temporal_lo, n_lo, cv::CMP_GE);
        cv::compare(d_prev, args.temporal_very_hi, p_very, cv::CMP_GE);

        cv::Mat a, b;
        cv::bitwise_and(p_hi, n_lo, a);
        cv::bitwise_and(n_hi, p_lo, b);
        cv::bitwise_or(a, b, temporal);
        cv::bitwise_or(temporal, p_very, temporal);
    }

    cv::Mat motion;
    cv::bitwise_and(temporal, old_raw, motion);

    if (args.close_size > 1) {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(args.close_size, args.close_size));
        cv::morphologyEx(motion, motion, cv::MORPH_CLOSE, kernel);
    }
    if (args.open_size > 1) {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(args.open_size, args.open_size));
        cv::morphologyEx(motion, motion, cv::MORPH_OPEN, kernel);
    }
    return component_filter(motion, args, counts, boxes);
}

cv::Mat overlay_frame(
    const cv::Mat& frame,
    const cv::Mat& old_raw,
    const cv::Mat& new_raw,
    const std::vector<cv::Rect>& boxes,
    const FrameStats& st,
    double live_fps
) {
    cv::Mat out = frame.clone();
    cv::Mat old_mask = old_raw > 0;
    cv::Mat new_mask = new_raw > 0;
    cv::Mat both, old_only, new_only, not_new, not_old;
    cv::bitwise_and(old_mask, new_mask, both);
    cv::bitwise_not(new_mask, not_new);
    cv::bitwise_not(old_mask, not_old);
    cv::bitwise_and(old_mask, not_new, old_only);
    cv::bitwise_and(new_mask, not_old, new_only);

    out.setTo(cv::Scalar(0, 0, 255), old_only);
    out.setTo(cv::Scalar(0, 255, 0), new_only);
    out.setTo(cv::Scalar(0, 255, 255), both);

    const std::size_t limit = std::min<std::size_t>(boxes.size(), 80);
    for (std::size_t i = 0; i < limit; ++i) {
        cv::rectangle(out, boxes[i], cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
    }

    cv::rectangle(out, cv::Rect(0, 0, std::min(out.cols, 860), 92), cv::Scalar(0, 0, 0), cv::FILLED);
    std::ostringstream line1;
    line1 << "C++ Raw Motion Probe | frame=" << st.frame
          << " | live=" << std::fixed << std::setprecision(1) << live_fps
          << " fps | motion=" << std::setprecision(2) << st.motion_ms << " ms";
    cv::putText(out, line1.str(), cv::Point(12, 24), cv::FONT_HERSHEY_SIMPLEX, 0.62,
                cv::Scalar(235, 235, 235), 2, cv::LINE_AA);

    std::ostringstream line2;
    line2 << "old raw px=" << st.old_raw_px << " | new raw px=" << st.new_raw_px
          << " | comps kept=" << st.kept_components << "/" << st.new_components;
    cv::putText(out, line2.str(), cv::Point(12, 52), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(220, 220, 220), 1, cv::LINE_AA);

    cv::putText(out, "red=old-only foreground/noise, yellow=overlap, green=new-only, cyan boxes=new kept components",
                cv::Point(12, 78), cv::FONT_HERSHEY_SIMPLEX, 0.50,
                cv::Scalar(210, 210, 210), 1, cv::LINE_AA);
    return out;
}

double mean_int(const std::vector<FrameStats>& rows, int FrameStats::*field) {
    if (rows.empty()) return 0.0;
    double sum = 0.0;
    for (const auto& r : rows) sum += static_cast<double>(r.*field);
    return sum / static_cast<double>(rows.size());
}

double mean_double(const std::vector<FrameStats>& rows, double FrameStats::*field) {
    if (rows.empty()) return 0.0;
    double sum = 0.0;
    for (const auto& r : rows) sum += r.*field;
    return sum / static_cast<double>(rows.size());
}

void write_json(
    const Args& args,
    const std::vector<FrameStats>& rows,
    double runtime_sec
) {
    std::ofstream f(args.json, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Could not open JSON output: " + args.json);
    }
    const double fps = static_cast<double>(rows.size()) / std::max(1e-9, runtime_sec);
    f << std::fixed << std::setprecision(6);
    f << "{\n";
    f << "  \"input\": \"" << json_escape(args.input) << "\",\n";
    f << "  \"output\": \"" << json_escape(args.output) << "\",\n";
    f << "  \"summary\": {\n";
    f << "    \"frames\": " << rows.size() << ",\n";
    f << "    \"runtime_sec\": " << runtime_sec << ",\n";
    f << "    \"effective_fps\": " << fps << ",\n";
    f << "    \"avg_old_raw_px\": " << mean_int(rows, &FrameStats::old_raw_px) << ",\n";
    f << "    \"avg_new_raw_px\": " << mean_int(rows, &FrameStats::new_raw_px) << ",\n";
    f << "    \"avg_motion_ms\": " << mean_double(rows, &FrameStats::motion_ms) << ",\n";
    f << "    \"avg_kept_components\": " << mean_int(rows, &FrameStats::kept_components) << "\n";
    f << "  },\n";
    f << "  \"frames\": [\n";
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        f << "    {\"frame\": " << r.frame
          << ", \"old_raw_px\": " << r.old_raw_px
          << ", \"new_raw_px\": " << r.new_raw_px
          << ", \"new_components\": " << r.new_components
          << ", \"kept_components\": " << r.kept_components
          << ", \"rejected_large\": " << r.rejected_large
          << ", \"rejected_shape\": " << r.rejected_shape
          << ", \"motion_ms\": " << r.motion_ms << "}";
        if (i + 1 != rows.size()) f << ",";
        f << "\n";
    }
    f << "  ]\n";
    f << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);

        cv::VideoCapture cap(args.input);
        if (!cap.isOpened()) {
            throw std::runtime_error("Could not open input video: " + args.input);
        }
        const double fps = cap.get(cv::CAP_PROP_FPS) > 0.0 ? cap.get(cv::CAP_PROP_FPS) : 30.0;
        const int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        const int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        const int total = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

        cv::VideoWriter writer(
            args.output,
            cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
            fps,
            cv::Size(width, height)
        );
        if (!writer.isOpened()) {
            throw std::runtime_error("Could not open output video: " + args.output);
        }

        cv::Mat prev_frame, curr_frame;
        if (!cap.read(prev_frame) || !cap.read(curr_frame)) {
            throw std::runtime_error("Input video needs at least two frames");
        }

        cv::Mat prev_hsv, curr_hsv, prev_gray, curr_gray;
        cv::cvtColor(prev_frame, prev_hsv, cv::COLOR_BGR2HSV);
        cv::cvtColor(curr_frame, curr_hsv, cv::COLOR_BGR2HSV);
        cv::cvtColor(prev_frame, prev_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(curr_frame, curr_gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Mat> init_ch;
        cv::split(prev_hsv, init_ch);
        cv::Mat bg_v, bg_s, var_v, var_s;
        init_ch[2].convertTo(bg_v, CV_32F);
        init_ch[1].convertTo(bg_s, CV_32F);
        var_v = cv::Mat(prev_frame.size(), CV_32F, cv::Scalar(args.motion_thresh * args.motion_thresh));
        var_s = cv::Mat(prev_frame.size(), CV_32F, cv::Scalar(args.motion_thresh * args.motion_thresh));

        std::vector<FrameStats> rows;
        rows.reserve(total > 0 ? static_cast<std::size_t>(total) : 2048);
        const auto run_start = Clock::now();
        int frame_idx = 1;

        while (true) {
            cv::Mat next_frame;
            const bool has_next = cap.read(next_frame);
            cv::Mat next_hsv, next_gray;
            if (has_next) {
                cv::cvtColor(next_frame, next_hsv, cv::COLOR_BGR2HSV);
                cv::cvtColor(next_frame, next_gray, cv::COLOR_BGR2GRAY);
            }

            const auto motion_start = Clock::now();
            cv::Mat old_raw = adaptive_sv_raw(curr_hsv, bg_v, bg_s, var_v, var_s, args);

            ComponentCounts counts;
            std::vector<cv::Rect> boxes;
            cv::Mat new_raw = new_raw_motion(
                prev_hsv,
                curr_hsv,
                has_next ? &next_hsv : nullptr,
                prev_gray,
                curr_gray,
                has_next ? &next_gray : nullptr,
                old_raw,
                args,
                counts,
                boxes
            );
            const double motion_ms = elapsed_ms(motion_start);

            FrameStats st;
            st.frame = frame_idx;
            st.old_raw_px = cv::countNonZero(old_raw);
            st.new_raw_px = cv::countNonZero(new_raw);
            st.new_components = counts.components;
            st.kept_components = counts.kept;
            st.rejected_large = counts.rejected_large;
            st.rejected_shape = counts.rejected_shape;
            st.motion_ms = motion_ms;
            rows.push_back(st);

            const double live_fps = static_cast<double>(frame_idx + 1) / std::max(1e-9, elapsed_sec(run_start));
            writer.write(overlay_frame(curr_frame, old_raw, new_raw, boxes, st, live_fps));

            update_background(curr_hsv, bg_v, bg_s, var_v, var_s, old_raw, args);

            ++frame_idx;
            if (args.max_frames > 0 && frame_idx >= args.max_frames) {
                break;
            }
            if (args.progress_every > 0 && frame_idx % args.progress_every == 0) {
                const double live = static_cast<double>(frame_idx) / std::max(1e-9, elapsed_sec(run_start));
                std::cout << "[raw-motion-cpp " << frame_idx << "/" << total << "] "
                          << std::fixed << std::setprecision(1) << live << " fps\n";
            }
            if (!has_next) {
                break;
            }

            prev_frame = curr_frame;
            curr_frame = next_frame;
            prev_hsv = curr_hsv;
            curr_hsv = next_hsv;
            prev_gray = curr_gray;
            curr_gray = next_gray;
        }

        writer.release();
        const double runtime_sec = elapsed_sec(run_start);
        write_json(args, rows, runtime_sec);
        std::cout << "[raw-motion-cpp] wrote video: " << args.output << "\n";
        std::cout << "[raw-motion-cpp] wrote json:  " << args.json << "\n";
        std::cout << "[raw-motion-cpp] done: frames=" << rows.size()
                  << " fps=" << (static_cast<double>(rows.size()) / std::max(1e-9, runtime_sec))
                  << " avg_motion_ms=" << mean_double(rows, &FrameStats::motion_ms) << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[raw-motion-cpp][error] " << e.what() << "\n";
        return 1;
    }
}
