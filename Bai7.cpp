#include <iostream>
#include <cstring>
using namespace std;

const int MAX = 100;

struct HoiThao {
    char chuDe[100];
    int batDau;
    int ketThuc;
};

// Hàm hoán đổi
void swap(HoiThao &a, HoiThao &b) {
    HoiThao temp = a;
    a = b;
    b = temp;
}

// Sắp xếp theo thời gian kết thúc tăng dần (selection sort)
void sapXep(HoiThao ds[], int n) {
    for (int i = 0; i < n - 1; ++i)
        for (int j = i + 1; j < n; ++j)
            if (ds[i].ketThuc > ds[j].ketThuc)
                swap(ds[i], ds[j]);
}

int main() {
    int n;
    HoiThao ds[MAX];

    cout << "Nhap so luong hoi thao: ";
    cin >> n;

    for (int i = 0; i < n; ++i) {
        cout << "Hoi thao " << i + 1 << ":\n";
        cout << "  Chu de: "; cin.ignore(); cin.getline(ds[i].chuDe, 100);
        cout << "  Thoi gian bat dau: "; cin >> ds[i].batDau;
        cout << "  Thoi gian ket thuc: "; cin >> ds[i].ketThuc;
    }

    // Sắp xếp theo thời gian kết thúc
    sapXep(ds, n);

    cout << "\nCac hoi thao sinh vien co the tham gia nhieu nhat:\n";
    int dem = 0;
    int thoiGianCuoi = -1;

    for (int i = 0; i < n; ++i) {
        if (ds[i].batDau >= thoiGianCuoi) {
            cout << "- " << ds[i].chuDe << " ("
                 << ds[i].batDau << " -> " << ds[i].ketThuc << ")\n";
            thoiGianCuoi = ds[i].ketThuc;
            dem++;
        }
    }

    cout << "Tong so hoi thao co the tham gia: " << dem << endl;

    return 0;
}
