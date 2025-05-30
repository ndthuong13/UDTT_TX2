#include <iostream>
#include <string>
#include <cstring>
using namespace std;

const int MAX = 10;
const int MAX_LEN = 101;

char d[MAX][MAX_LEN]; // danh sách xâu

// Hàm hỗ trợ 1: kiểm tra xâu str có chứa ít nhất một xâu trong danh sách d
bool chuaMotXau(char str[], int n) {
    for (int i = 0; i < n; ++i) {
        if (strstr(str, d[i]) != NULL)
            return true;
    }
    return false;
}

// Hàm hỗ trợ 2: Tham lam chọn k ký tự từ d sao cho chứa ít nhất một xâu d[i]
void thamLam(int n, int k) {
    char result[101] = "";
    int len = 0;
    for (int i = 0; i < n && len < k; ++i) {
        for (int j = 0; d[i][j] && len < k; ++j) {
            result[len++] = d[i][j];
        }
    }
    result[k] = '\0';
    if (chuaMotXau(result, n)) {
        cout << "Xau tao thanh: " << result << "\n";
    } else {
        cout << "Khong tao duoc xau thoa dieu kien\n";
    }
}

// Hàm hỗ trợ 3: Boyer Moore Horspool cho từ "child"
int BMH_count(const char* text, const char* pattern) {
    int n = strlen(text);
    int m = strlen(pattern);
    if (m > n) return 0;

    int badChar[256];
    for (int i = 0; i < 256; ++i)
        badChar[i] = m;

    for (int i = 0; i < m - 1; ++i)
        badChar[(int)pattern[i]] = m - 1 - i;

    int count = 0, i = 0;
    while (i <= n - m) {
        int j = m - 1;
        while (j >= 0 && pattern[j] == text[i + j]) j--;
        if (j < 0) {
            count++;
            i += m;
        } else {
            i += badChar[(int)text[i + m - 1]];
        }
    }
    return count;
}

void BMH_findChild(int n) {
    const char* keyword = "child";
    cout << "Xau chua 'child':\n";
    for (int i = 0; i < n; ++i) {
        int cnt = BMH_count(d[i], keyword);
        if (cnt > 0)
            cout << "- " << d[i] << " (" << cnt << " lan)\n";
    }
}

// Hàm hỗ trợ 4: Tính mảng Z cho Z-Algorithm
void computeZ(char s[], int Z[]) {
    int n = strlen(s);
    int L = 0, R = 0;
    Z[0] = n;
    for (int i = 1; i < n; ++i) {
        if (i > R) {
            L = R = i;
            while (R < n && s[R - L] == s[R])
                R++;
            Z[i] = R - L;
            R--;
        } else {
            int k = i - L;
            if (Z[k] < R - i + 1)
                Z[i] = Z[k];
            else {
                L = i;
                while (R < n && s[R - L] == s[R])
                    R++;
                Z[i] = R - L;
                R--;
            }
        }
    }
}

void Z_search(int n) {
    char* pattern = d[0];
    int lenP = strlen(pattern);

    cout << "\nXau chua d[0]:\n";
    for (int i = 1; i < n; ++i) {
        int lenT = strlen(d[i]);
        char concat[MAX_LEN * 2];
        strcpy(concat, pattern);
        strcat(concat, "$");
        strcat(concat, d[i]);

        int Z[MAX_LEN * 2] = {};
        computeZ(concat, Z);

        int count = 0;
        for (int j = lenP + 1; j < strlen(concat); ++j) {
            if (Z[j] == lenP) count++;
        }

        if (count > 0)
            cout << "- " << d[i] << ": " << count << " lan\n";
    }
}

int main() {
    int n;
    cout << "Nhap so xau n (5 <= n <= 10): ";
    cin >> n;
    cin.ignore();

    for (int i = 0; i < n; ++i) {
        cout << "Nhap xau thu " << i << ": ";
        cin.getline(d[i], MAX_LEN);
    }

    int k;
    cout << "\nNhap do dai xau can tao (k): ";
    cin >> k;

    cout << "\n--- Tham lam ---\n";
    thamLam(n, k);

    cout << "\n--- Boyer Moore Horspool ---\n";
    BMH_findChild(n);

    cout << "\n--- Z Algorithm ---\n";
    Z_search(n);

    return 0;
}
