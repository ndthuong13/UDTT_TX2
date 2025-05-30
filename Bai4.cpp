#include <iostream>
#include <string>
using namespace std;

const int MAX = 10;     // tối đa 10 điện thoại
const int SCALE = 10;   // scale cho float -> int

struct DienThoai {
    string nhanHieu;
    float kichThuoc;  // inch
    int giaBan;
};

int main() {
    DienThoai d[MAX];
    int n;
    float s;

    cout << "Nhap kich thuoc tui (inch): ";
    cin >> s;
    cout << "Nhap so luong dien thoai n (0 < n <= 10): ";
    cin >> n;

    cin.ignore();
    for (int i = 0; i < n; ++i) {
        cout << "\nDien thoai " << i + 1 << ":\n";
        cout << "- Nhan hieu: ";
        getline(cin, d[i].nhanHieu);
        cout << "- Kich thuoc man hinh (inch): ";
        cin >> d[i].kichThuoc;
        cout << "- Gia ban: ";
        cin >> d[i].giaBan;
        cin.ignore();
    }

    // Chuyển s sang đơn vị int (scale x10)
    int capacity = (int)(s * SCALE + 0.5);
    int dp[MAX + 1][MAX * 100 + 1] = {}; // dp[i][j]: xét i thiết bị, dung lượng j
    bool chon[MAX + 1][MAX * 100 + 1] = {}; // lưu truy vết chọn hay không

    // Lập bảng quy hoạch động
    for (int i = 1; i <= n; ++i) {
        int weight = (int)(d[i - 1].kichThuoc * SCALE + 0.5);
        int value = d[i - 1].giaBan;
        for (int j = 0; j <= capacity; ++j) {
            if (weight <= j && dp[i - 1][j - weight] + value > dp[i - 1][j]) {
                dp[i][j] = dp[i - 1][j - weight] + value;
                chon[i][j] = true;
            } else {
                dp[i][j] = dp[i - 1][j];
                chon[i][j] = false;
            }
        }
    }

    // Truy vết ngược để in danh sách điện thoại đã chọn
    int j = capacity;
    DienThoai ketQua[MAX];
    int dem = 0;

    for (int i = n; i >= 1; --i) {
        int weight = (int)(d[i - 1].kichThuoc * SCALE + 0.5);
        if (chon[i][j]) {
            ketQua[dem++] = d[i - 1];
            j -= weight;
        }
    }

    cout << "\nCac dien thoai duoc chon:\n";
    int tongGia = 0;
    for (int i = dem - 1; i >= 0; --i) {
        cout << "- " << ketQua[i].nhanHieu << ", Gia ban: " << ketQua[i].giaBan << endl;
        tongGia += ketQua[i].giaBan;
    }

    cout << "Tong gia tri: " << tongGia << endl;

    return 0;
}
