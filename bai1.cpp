#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

using namespace std;

struct Quat {
    string tenHang;
    string mauSac;
    int giaBan;
};

// Hàm so sánh dùng cho sort (giá tăng dần)
bool soSanhGia(const Quat &a, const Quat &b) {
    return a.giaBan < b.giaBan;
}

int main() {
    int p, n;
    cout << "Nhap so tien p (0 < p < 1200000): ";
    cin >> p;
    cout << "Nhap so luong quat n (5 < n < 12): ";
    cin >> n;

    vector<Quat> dsQuat(n);
    cout << "Nhap thong tin tung quat:\n";
    for (int i = 0; i < n; ++i) {
        cout << "Quat " << i + 1 << ":\n";
        cout << "- Ten hang san xuat: ";
        cin.ignore(); // Xóa bộ nhớ đệm trước khi getline
        getline(cin, dsQuat[i].tenHang);
        cout << "- Mau sac: ";
        getline(cin, dsQuat[i].mauSac);
        cout << "- Gia ban: ";
        cin >> dsQuat[i].giaBan;
    }

    // Sắp xếp danh sách theo giá bán tăng dần
    sort(dsQuat.begin(), dsQuat.end(), soSanhGia);

    vector<Quat> daMua;
    int tongTien = 0;

    for (const auto &quat : dsQuat) {
        if (tongTien + quat.giaBan <= p) {
            daMua.push_back(quat);
            tongTien += quat.giaBan;
        } else {
            break;
        }
    }

    // Kết quả
    cout << "\nSo quat co the mua: " << daMua.size() << endl;
    cout << "Danh sach cac quat da mua:\n";
    for (const auto &quat : daMua) {
        cout << "- Hang: " << quat.tenHang << ", Gia ban: " << quat.giaBan << endl;
    }

    return 0;
}
