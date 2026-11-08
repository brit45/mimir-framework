#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

namespace mimir::viz {

struct CommandCardRect {
    float x = 0.f, y = 0.f, width = 0.f, height = 0.f;
};

struct CommandButtonLayout {
    CommandCardRect bounds;
    std::vector<std::string> lines;
    unsigned text_size = 12;
    // Glyph positions relative to the button; the renderer removes font bearings.
    float text_x = 12.f, text_y = 9.f, line_height = 16.f;
};

struct CommandCardLayout {
    std::array<CommandButtonLayout, 10> buttons;
    std::vector<std::string> primary_hint_lines;
    float primary_hint_y = 0.f; // Relative to the card, like status_y.
    std::vector<std::string> status_lines;
    float status_y = 0.f, status_line_height = 15.f, height = 0.f;
};

// Measure returns the glyph width of a UTF-8 string at the requested size/style.
// Text is wrapped at words, or complete UTF-8 code points for a long path/token.
template <class Measure>
std::vector<std::string> wrapCommandCardText(const std::string& value, float width,
                                           unsigned size, bool bold, Measure measure) {
    std::vector<std::string> lines;
    std::istringstream paragraphs(value);
    for (std::string paragraph; std::getline(paragraphs, paragraph);) {
        std::istringstream words(paragraph);
        std::string line, word;
        while (words >> word) {
            const std::string candidate = line.empty() ? word : line + " " + word;
            if (measure(candidate, size, bold) <= width) { line = candidate; continue; }
            if (!line.empty()) { lines.push_back(std::move(line)); line.clear(); }
            if (measure(word, size, bold) <= width) { line = word; continue; }
            for (std::size_t offset = 0; offset < word.size();) {
                const auto lead = static_cast<unsigned char>(word[offset]);
                const std::size_t count = std::min(word.size() - offset,
                    std::size_t(lead < 0x80 ? 1 : (lead & 0xe0) == 0xc0 ? 2 :
                                (lead & 0xf0) == 0xe0 ? 3 : 4));
                const auto character = word.substr(offset, count);
                if (!line.empty() && measure(line + character, size, bold) > width) {
                    lines.push_back(std::move(line)); line.clear();
                }
                line += character;
                offset += count;
            }
        }
        if (!line.empty() || paragraph.empty()) lines.push_back(std::move(line));
    }
    return lines;
}

template <class Measure>
CommandCardLayout makeCommandCardLayout(float width,
                                       const std::array<std::string, 10>& labels,
                                       const std::string& primary_hint,
                                       const std::string& status,
                                       Measure measure) {
    constexpr float padding = 8.f, gap = 8.f;
    const float inner_width = std::max(1.f, width - 2.f * padding);
    CommandCardLayout layout;
    auto& primary = layout.buttons[0];
    primary.text_size = 14;
    primary.text_x = 40.f;
    primary.text_y = 8.f;
    primary.line_height = 18.f;
    const float primary_text_width = std::max(1.f, inner_width - primary.text_x - 10.f);
    primary.lines = wrapCommandCardText(labels[0], primary_text_width, 14, true, measure);
    layout.primary_hint_lines = wrapCommandCardText(primary_hint, primary_text_width, 10, false, measure);
    const float primary_label_height = primary.line_height * primary.lines.size();
    const float primary_height = std::max(48.f, 8.f + primary_label_height + 3.f +
        12.f * layout.primary_hint_lines.size() + 7.f);
    primary.bounds = {padding, 28.f, inner_width, primary_height};
    layout.primary_hint_y = primary.bounds.y + primary.text_y + primary_label_height + 3.f;
    float next_y = primary.bounds.y + primary_height + gap;

    auto addRows = [&](std::size_t first, std::size_t count, int max_columns,
                       float minimum_width) {
        float preferred_width = minimum_width;
        for (std::size_t i = first; i < first + count; ++i)
            preferred_width = std::max(preferred_width, measure(labels[i], 12, true) + 24.f);
        const int columns = std::clamp(static_cast<int>((inner_width + gap) /
            (preferred_width + gap)), 1, max_columns);
        const float button_width = (inner_width - (columns - 1) * gap) / columns;
        for (std::size_t row_start = first; row_start < first + count; row_start += columns) {
            const std::size_t row_count = std::min<std::size_t>(columns, first + count - row_start);
            // Give the last action the full remaining row instead of leaving a narrow cell.
            const float row_width = row_count == 1 ? inner_width : button_width;
            float row_height = 34.f;
            for (std::size_t column = 0; column < row_count; ++column) {
                auto& button = layout.buttons[row_start + column];
                button.lines = wrapCommandCardText(labels[row_start + column],
                    std::max(1.f, row_width - 24.f), 12, true, measure);
                row_height = std::max(row_height, 16.f * button.lines.size() + 16.f);
            }
            for (std::size_t column = 0; column < row_count; ++column) {
                auto& button = layout.buttons[row_start + column];
                button.bounds = {padding + column * (button_width + gap), next_y,
                                 row_width, row_height};
                button.text_y = (row_height - button.line_height * button.lines.size()) * .5f;
            }
            next_y += row_height + gap;
        }
    };
    addRows(1, 6, 3, 132.f);
    // Long actions use whole rows until two complete labels fit at 12 px.
    addRows(7, 3, 2, 180.f);
    layout.status_lines = wrapCommandCardText(status, inner_width, 11, false, measure);
    layout.status_y = next_y;
    layout.height = next_y + layout.status_line_height * layout.status_lines.size() + padding;
    return layout;
}

} // namespace mimir::viz
