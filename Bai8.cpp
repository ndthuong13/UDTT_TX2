#include <iostream>
#include <cstring>
using namespace std;

const int MAX = 100;

struct Laptop {
    char hang[50];
    char cauHinh[200];
    float gia;
};

// Thuật toán Boyer Moore Horspool (tìm vị trí đầu tiên của pattern trong text)
int BMP(const char* text, const char* pattern) {
    int n = strlen(text);
    int m = strlen(pattern);
    if (m > n) return -1;

    int shift[256];
    for (int i = 0; i < 256; ++i) shift[i] = m;
    for (int i = 0; i < m - 1; ++i)
        shift[(unsigned char)pattern[i]] = m - 1 - i;

    int i = 0;
    while (i <= n - m) {
        int j = m - 1;
        while (j >= 0 && pattern[j] == text[i + j]) j--;
        if (j < 0) return i;
        i += shift[(unsigned char)text[i + m - 1]];
    }
    return -1;
}

// Thuật toán Z: Kiểm tra pattern có xuất hiện trong text hay không
bool ZSearch(const char* pattern, const char* text) {
    char concat[400];
    strcpy(concat, pattern);
    strcat(concat, "$");
    strcat(concat, text);

    int n = strlen(concat);
    int m = strlen(pattern);
    int Z[400] = {0};

    int L = 0, R = 0;
    for (int i = 1; i < n; ++i) {
        if (i > R) {
            L = R = i;
            while (R < n && concat[R - L] == concat[R]) R++;
            Z[i] = R - L;
            R--;
        } else {
            int k = i - L;
            if (Z[k] < R - i + 1)
                Z[i] = Z[k];
            else {
                L = i;
                while (R < n && concat[R - L] == concat[R]) R++;
                Z[i] = R - L;
                R--;
            }
        }
    }

    for (int i = 0; i < n; ++i)
        if (Z[i] == m) return true;

    return false;
}

int main() {
    int n;
    Laptop ds[MAX];

    cout << "Nhap so luong laptop: ";
    cin >> n;
    cin.ignore();

    for (int i = 0; i < n; ++i) {
        cout << "Laptop " << i + 1 << ":\n";
        cout << "  Hang SX: "; cin.getline(ds[i].hang, 50);
        cout << "  Cau hinh: "; cin.getline(ds[i].cauHinh, 200);
        cout << "  Gia ban: "; cin >> ds[i].gia; cin.ignore();
    }

    // Tìm laptop có RAM 16GB
    int demRAM = 0;
    for (int i = 0; i < n; ++i) {
        if (BMP(ds[i].cauHinh, "RAM 16GB") != -1)
            demRAM++;
    }

    cout << "\nSo laptop co RAM 16GB: " << demRAM << endl;

    // Hiển thị laptop dùng SSD
    cout << "Laptop dung SSD:\n";
    bool found = false;
    for (int i = 0; i < n; ++i) {
        if (ZSearch("SSD", ds[i].cauHinh)) {
            found = true;
            cout << "- Hang: " << ds[i].hang
                 << ", Cau hinh: " << ds[i].cauHinh
                 << ", Gia: " << ds[i].gia << " VND\n";
        }
    }

    if (!found)
        cout << "Khong co laptop nao dung SSD.\n";

    return 0;
}
