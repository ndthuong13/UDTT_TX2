// NguyenVanA_12345678.CPP
#include <bits/stdc++.h>
using namespace std;

struct ManHinh {
    string hangSanXuat;
    float kichThuoc;
    float giaBan;
};

// Khởi tạo dữ liệu danh sách n màn hình (n >= 7)
void khoiTaoDanhSach(vector<ManHinh>& ds) {
    ds = {
        {"LG", 15.6, 500},
        {"HP", 14.0, 450},
        {"Samsung", 17.0, 600},
        {"Asus", 13.3, 400},
        {"Acer", 15.6, 480},
        {"Dell", 16.0, 550},
        {"Lenovo", 15.0, 490}
    };
}

// A1: Đệ quy tính tổng giá bán
float tongGiaBan(const vector<ManHinh>& ds, int i = 0) {
    if (i == ds.size()) return 0;
    return ds[i].giaBan + tongGiaBan(ds, i + 1);
}

// A2: Đệ quy đếm số lượng màn hình >= 15.6 inch
int demLonHon156(const vector<ManHinh>& ds, int i = 0) {
    if (i == ds.size()) return 0;
    return (ds[i].kichThuoc >= 15.6 ? 1 : 0) + demLonHon156(ds, i + 1);
}

// Hiển thị danh sách hãng theo thứ tự 1...n
void hienThiDanhSach(const vector<ManHinh>& ds) {
    for (int i = 0; i < ds.size(); ++i) {
        cout << i + 1 << " - " << ds[i].hangSanXuat << endl;
    }
}

// A3: Sinh hoán vị đệ quy và hiển thị các phương án
void sinhHoanVi(vector<ManHinh>& ds, int l, int r, int& dem) {
    if (l == r) {
        cout << "Phuong an " << ++dem << ": ";
        for (auto mh : ds) cout << mh.hangSanXuat << " ";
        cout << endl;
        return;
    }
    for (int i = l; i <= r; ++i) {
        swap(ds[l], ds[i]);
        sinhHoanVi(ds, l + 1, r, dem);
        swap(ds[l], ds[i]); // quay lui
    }
}

int main() {
    vector<ManHinh> danhSach;
    khoiTaoDanhSach(danhSach);

    cout << fixed << setprecision(2);

    // A1
    float tong = tongGiaBan(danhSach);
    cout << "Tong gia ban: " << tong << endl;

    // A2
    int soLuong = demLonHon156(danhSach);
    cout << "So man hinh >= 15.6 inch: " << soLuong << endl;
    cout << "Danh sach hang san xuat theo thu tu:\n";
    hienThiDanhSach(danhSach);

    // A3
    int dem = 0;
    cout << "Cac phuong an xep man hinh (thu tu):\n";
    sinhHoanVi(danhSach, 0, danhSach.size() - 1, dem);
    cout << dem << endl;

    return 0;
}
