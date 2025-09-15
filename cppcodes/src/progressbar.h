#pragma once
/*
ANSI转议序列
    "\x1b[?7h",//当字符显示到行末时，自动回到下行行首接着显示；如果在滚动区域底行行末，则上滚一行再显示
    "\x1b[?7l",//当字符显示到行末时，仍在行末光标位置显示，覆盖原有的字符，除非接收到移动光标的命令
    "\x1b[K",//清除光标至行末字符，包括光标位置，行属性不受影响。
    "\x1b[0K",//清除光标至行末字符，包括光标位置，行属性不受影响。
    "\x1b[1K",//清除行首至光标位置字符，包括光标位置，行属性不受影响。
    "\x1b[2K",//清除光标所在行的所有字符

    \033[30m  # 设置前景色为黑色
    \033[31m  # 设置前景色为红色
    \033[32m  # 设置前景色为绿色
    \033[33m  # 设置前景色为黄色
    \033[34m  # 设置前景色为蓝色
    \033[35m  # 设置前景色为品红
    \033[36m  # 设置前景色为青色
    \033[37m  # 设置前景色为白色

    \033[40m  # 设置背景色为黑色
    \033[41m  # 设置背景色为红色
    \033[42m  # 设置背景色为绿色
    \033[43m  # 设置背景色为黄色
    \033[44m  # 设置背景色为蓝色
    \033[45m  # 设置背景色为品红
    \033[46m  # 设置背景色为青色
    \033[47m  # 设置背景色为白色
*/


#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>

#include <chrono>

#if 0
#if defined(WIN32)
#include <windows.h>

void getTerminalSize(int& rows, int& cols) {
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
    rows = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;
    cols = csbi.srWindow.Right - csbi.srWindow.Left + 1;
}
#else
#include <termios.h>
#include <sys/ioctl.h>

void getTerminalSize(int& rows, int& cols) {
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);       //STDOUT_FILENO c20
    rows = w.ws_row;
    cols = w.ws_col;
}
#endif
#endif

class progressbar{
public:
    progressbar(){
        progressbar(100, 25);
    };
    /*
    * total progress bar 对应计数器的最大值
    * barwidth = 25, 用多少字符来性展示progress
    * 
    */
    progressbar(int total, int barwidth = 25)
    {
        total_num = total;
        bar_width = barwidth;

        progress = 0;
        prefix = "";
        suffix = "";

        todo_char = " ";
        done_char = "#";

        completion_rate = 0.f;

        start_tp = std::chrono::system_clock::now();
    }
    ~progressbar() = default;

    void set_prefix(const std::string& str) { prefix = str; }
    void set_suffix(const std::string& str) { suffix = str; }
    void set_todo_char(const std::string& str) { todo_char = str;}
    void set_done_char(const std::string& str) { done_char = str; }
    void update()
    {
        progress += 1;

        auto now_tp = std::chrono::system_clock::now();
        
        auto diff_start = std::chrono::duration_cast<std::chrono::seconds>(now_tp - start_tp).count();
        auto diff = (diff_start * total_num)/progress;

        auto get_time_fmtstr = [](long long dt) {
            auto h = dt / (60 * 60);
            auto m = (dt % (60 * 60)) / 60;
            auto s = dt % 60;

            std::stringstream ss;

            ss << std::setw(3) << h << ":" << std::setw(2) << std::setfill('0')<< m << ":" << std::setw(2) << std::setfill('0') <<s;
            return ss.str();
            };
    
        completion_rate = float(progress) / float(total_num);
        auto done_len = std::min(int(completion_rate * bar_width), bar_width);
        auto todo_len = bar_width - done_len;

        std::stringstream ss_bar;
        ss_bar << "[";
        for (int i = 0; i < done_len; i++) ss_bar << done_char;
        for (int i = 0; i < todo_len; i++) ss_bar << todo_char;
        ss_bar << "] " << std::min(100,int(completion_rate * 100.f)) << "% ";

        auto st1 = get_time_fmtstr(diff_start);
        auto st2 = get_time_fmtstr(diff);

        ss_bar << " " << st1 << "|" << st2;

        std::stringstream ss_progresses;
        ss_progresses << prefix << " " << ss_bar.str() << suffix;

        std::cout << "\x1b[2K\r" << ss_progresses.str();
        std::cout << std::flush;        
    }

private:
    int total_num;
    int bar_width;
    int progress;       //当前进度

    std::string prefix; 
    std::string suffix;

    std::string todo_char;
    std::string done_char;

    float completion_rate;    // 完成率

    std::chrono::system_clock::time_point start_tp;
    std::chrono::system_clock::time_point last_tp;
};
