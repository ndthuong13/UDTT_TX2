#include <iostream>
using namespace std;

const int MAX = 100;

int a[MAX], b[MAX], dp[MAX + 1][MAX + 1];

int main() {
    int n, m;
    cout << "Nhap so phan tu cua a (n): ";
    cin >> n;
    cout << "Nhap day a: ";
    for (int i = 1; i <= n; ++i)
        cin >> a[i];

    cout << "Nhap so phan tu cua b (m): ";
    cin >> m;
    cout << "Nhap day b: ";
    for (int i = 1; i <= m; ++i)
        cin >> b[i];

    // Bước 1: Tính bảng dp
    for (int i = 0; i <= n; ++i)
        for (int j = 0; j <= m; ++j)
            dp[i][j] = 0;

    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            if (a[i] == b[j])
                dp[i][j] = dp[i - 1][j - 1] + 1;
            else
                dp[i][j] = (dp[i - 1][j] > dp[i][j - 1]) ? dp[i - 1][j] : dp[i][j - 1];
        }
    }

    // Bước 2: Truy vết để tìm dãy con chung dài nhất
    int i = n, j = m;
    int c[MAX], length = 0;

    while (i > 0 && j > 0) {
        if (a[i] == b[j]) {
            c[length++] = a[i];
            i--; j--;
        } else if (dp[i - 1][j] > dp[i][j - 1]) {
            i--;
        } else {
            j--;
        }
    }

    // Đảo ngược dãy kết quả
    cout << "Do dai day con chung dai nhat: " << length << endl;
    cout << "Day con chung: ";
    for (int k = length - 1; k >= 0; --k)
        cout << c[k] << " ";
    cout << endl;

    return 0;
}
