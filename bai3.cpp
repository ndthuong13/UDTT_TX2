#include <iostream>
#include <vector>
#include <string>
#include <climits>

using namespace std;

struct Quat {
    string tenHang;
    string mauSac;
    int giaBan;
};

int p, n;
vector<Quat> dsQuat;
vector<Quat> ketQuaToiUu;
int soLuongItNhat = INT_MAX;

void quayLui(int chiSo, int tong, vector<Quat> tapHienTai) {
    if (tong > p && tapHienTai.size() < soLuongItNhat) {
        soLuongItNhat = tapHienTai.size();
        ketQuaToiUu = tapHienTai;
        return;
    }
    
    if (chiSo >= n || tapHienTai.size() >= soLuongItNhat) return;

    // Nhánh chọn quạt tại chiSo
    tapHienTai.push_back(dsQuat[chiSo]);
    quayLui(chiSo + 1, tong + dsQuat[chiSo].giaBan, tapHienTai);
    tapHienTai.pop_back();

    // Nhánh không chọn
    quayLui(chiSo + 1, tong, tapHienTai);
}

int main() {
    cout << "Nhap so tien p (0 < p < 1200000): ";
    cin >> p;
    cout << "Nhap so luong quat n (5 < n < 12): ";
    cin >> n;

    dsQuat.resize(n);
    cin.ignore();

    for (int i = 0; i < n; ++i) {
        cout << "Quat " << i + 1 << ":\n";
        cout << "- Ten hang san xuat: ";
        getline(cin, dsQuat[i].tenHang);
        cout << "- Mau sac: ";
        getline(cin, dsQuat[i].mauSac);
        cout << "- Gia ban: ";
        cin >> dsQuat[i].giaBan;
        cin.ignore();
    }

    vector<Quat> tapHienTai;
    quayLui(0, 0, tapHienTai);

    if (ketQuaToiUu.empty()) {
        cout << "Khong co cach chon nao de tong gia lon hon " << p << endl;
    } else {
        cout << "\nSo quat it nhat de tong gia > " << p << ": " << ketQuaToiUu.size() << endl;
        cout << "Danh sach cac quat:\n";
        for (const auto &q : ketQuaToiUu) {
            cout << "- Hang: " << q.tenHang << ", Gia ban: " << q.giaBan << endl;
        }
    }

    return 0;
}
