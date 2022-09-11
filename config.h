#pragma once

#define ENABLE_GPU 1
constexpr auto FB_WIDTH = 160;
constexpr auto FB_HEIGHT = 144;
constexpr auto EMU_WIDTH = 8;
constexpr auto EMU_HEIGHT = 4;
constexpr auto WINDOW_WIDTH = FB_WIDTH * EMU_WIDTH;
constexpr auto WINDOW_HEIGHT = FB_HEIGHT * EMU_HEIGHT;
