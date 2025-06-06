#include <iostream>
using namespace std;

struct Balo {
    string nh;
    int kg;     
    int gt;    
};


void sapXepGiamTheoKg(Balo d[], int n) {
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (d[i].kg < d[j].kg) {
                Balo temp = d[i];
                d[i] = d[j];
                d[j] = temp;
            }
}


void thuatToanG(Balo d[], int n, int m, int &u, int &v, int &tongKg) {
    u = 0;
    v = 0;
    tongKg = 0;
    for (int i = 0; i < n; i++) {
        v += d[i].gt;
        tongKg += d[i].kg;
        u++;
        if (v > m) return;
    }
  
    u = 0; v = 0; tongKg = 0;
}


void thuatToanD(Balo d[], int n, int y, int q[], int &qSize, int &tongGT, int &tongKg) {
    int dp[n + 1][y + 1];

    for (int i = 0; i <= n; i++)
        for (int w = 0; w <= y; w++)
            dp[i][w] = 0;

    for (int i = 1; i <= n; i++) {
        for (int w = 0; w <= y; w++) {
            dp[i][w] = dp[i - 1][w];
            if (w >= d[i - 1].kg) {
                int val = dp[i - 1][w - d[i - 1].kg] + d[i - 1].gt;
                if (val > dp[i][w]) {
                    dp[i][w] = val;
                }
            }
        }
    }


    tongGT = 0;
    tongKg = 0;
    qSize = 0;
    int w = y;
    for (int i = n; i > 0 && w > 0; i--) {
        if (dp[i][w] != dp[i - 1][w]) {
            q[qSize++] = i - 1;
            tongGT += d[i - 1].gt;
            tongKg += d[i - 1].kg;
            w -= d[i - 1].kg;
        }
    }
}

int main() {
    int n;
    do {
        cout << "Nhap so luong ba lo n (>=7): ";
        cin >> n;
    } while (n < 7);

    Balo d[n];
    cout << "Nhap thong tin cho " << n << " ba lo:\n";
    for (int i = 0; i < n; i++) {
        cout << "- Ba lo thu " << i + 1 << ":\n";
        cout << "  Nhan hieu: "; cin >> d[i].nh;
        cout << "  Khoi luong: "; cin >> d[i].kg;
        cout << "  Gia tri: "; cin >> d[i].gt;
    }

    int m;
    cout << "\nNhap tong gia tri m (tham lam G can vuot qua): ";
    cin >> m;

    int y;
    cout << "Nhap khoi luong toi da y (quy hoach dong D): ";
    cin >> y;

    sapXepGiamTheoKg(d, n);

    int u, v, tongKg;
    thuatToanG(d, n, m, u, v, tongKg);
    if (u == 0)
        cout << "\n[Tham lam - G] Khong tim duoc phuong an thoa dieu kien!\n";
    else {
        cout << "\n[Tham lam - G] So ba lo can chon: " << u << ", Tong gia tri: " << v << ", Tong khoi luong: " << tongKg << "\n";
    }

    int q[n], qSize, tongGT_D, tongKg_D;
    thuatToanD(d, n, y, q, qSize, tongGT_D, tongKg_D);
    if (qSize == 0)
        cout << "\n[Quy hoach dong - D] Khong tim duoc phuong an phu hop!\n";
    else {
        cout << "\n[Quy hoach dong - D] Tong gia tri lon nhat: " << tongGT_D
             << ", Tong khoi luong: " << tongKg_D << "\n";
        cout << "Cac ba lo duoc chon:\n";
        for (int i = 0; i < qSize; i++) {
            int idx = q[i];
            cout << " - " << d[idx].nh << " (kg = " << d[idx].kg << ", gt = " << d[idx].gt << ")\n";
        }
    }

    return 0;
}
