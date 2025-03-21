# ��������� ��� ��������� ����������� � �������������� ��� runme.py

**�������� ���������**:  
������ ��������� ������������� ��� ��������� ����������� � ����� �����������, ������������� � ������� ���. ��� ���������� ���������� **OpenCV** ��� ������ � ������������� � �������������, � ����� �������������� ������ ��� ������ � ����� ������ ��� � ��������� ����������.

---

## �������� �������

### **������������� � ���������**:
- ��������� �������� � ������ ��������� � ������ � ������������� ����������� �������.
- ������������ ������ `PinFace` ��� ������ � ������, ������� ������������ ��������� ������ ����������� (`ffmode`) � ������������� (`frmode`).

### **�������� ������**:
- ����������� ��������� ���� �� ���� ������ (`face_data`).
- ������������� ���������������� ��������� �� ������ `config`.

### **��������� �����������**:
- ��������� ������ ���������� � ������������ ������ ����.
- ��� ������� ����� ����������� ����������� ���, �� ������������� � ������.

### **������������� ���**:
- ��� ������� ������������� ���� ����������� ��������� �������������.
- ������������ ����� � ���� ������ ��������� ��� � ���������� ���������� �� ��������� ����������.
- ���� ���� �� ����������, ��� ����������� � ����� `unknown`.

### **����������� �����������**:
- �� ����������� ������������� ���������� � ������������ �����, ������� ���, ������� ���������� � ������ ������.
- ���������� ������������ � ���� **OpenCV**.

### **�����������**:
- � ������� ��������� ��������� ������ � ������ ������������ �����, ������� ����� ���������� �������� � ���������� �������������.

---

## ������������ ����������

- **[OpenCV](https://docs.opencv.org/)**: ��� ������ � ������������� � �������������.
- **[NumPy](https://numpy.org/doc/)**: ��� ������ � ��������� ������.
- **���� ������ ���**: ��� �������� � ������ ��������� ���.
- **���������������� �����**: ��� ��������� ���������� ���������.

---

## ��� ������������

### ��������� ������������

���������, ��� ����������� ��� ����������� ����������:

```bash
pip install opencv-python numpy
```

## ��������� ������������

�������������� ���� `config.py`, ����� ������� ��������� �����������, ���� � ������ � ������ ���������.

---

## ������ ���������

��������� ������:

```bash
python runme.py
```

��������� ������ ��������� ����������� � ��������� ���������� � ���� **OpenCV**.

---

## ����������

��� ������ �� ��������� ������� ������� `q`.

---


# readphoto.py

������ ������ ������������ ��� ��������� ����������� � ������, �� ������� � ���������� � ���� ������. ��������� ���������� ���������� ��� ������ � ������������� � ������������ �������, ����� ��� `Pillow`, `OpenCV`, `numpy`, � ����� ����������� ������ ��� ����������� � ������������� ���.

---

## �������� �������

### 1. **������������� � ���������**
- ��������� �������� ������ � ������ ������ � ������������� ����������� �������.
- ������������ ������ ����������� (`ffmode`) � ������������� (`frmode`) ���.
- ���������������� ������ `PinFace` ��� ������ � ������.

### 2. **�������� ������**
- ����������� ��������� ���� �� ���� ������ (`known_file`, `known_vector`).
- ��������� ���� ��� ����� � ����������� `.jp*` (��������, `.jpg`, `.jpeg`) � ��������� ����������.

### 3. **��������� �����������**
- ��� ������� �����������:
  - �����������, �������� �� ���� ���������� (�� ������ ����� � �������).
  - ���� ���� ��� ���������, �� ������������ � ����� `/double/`.
  - ���� ���� �� ����� ���� ��������, �� ������������ � ����� `/error/`.
  - ����������� ����������� � ������������� � RGB.
  - �� ����������� �������������� ���� � ������� `PinFace.face_detection`.
  - ���� ��� �� �������, ���� ������������ � ����� `/empty/`.
  - ���� ������� ��������� ���, ���� ������������ � ����� `/many/`.
  - ���� ������� ���� ����, ��� �������������� � ����������� � ���� ������.

### 4. **������������� ���**
- ��� ������� ������������� ����:
  - ����������� ���������� (��������� �������������) � �������������� ������� `sface` � `adaface`.
  - ��� ����� �������������� ��� ���������� ����� ��������.
  - ������ ����������� � ���� ������, � ���� ������������ � ����� `/ok/`.

### 5. **���������� ������**
- ��������� ������� ���������� �� ����������� �������.
- ���� ������ �� �����������, ��������� ��������������� ���������.
- ���� ������ ���� ���������, ��������� ���� `newrecods.flag` � ����������� � ���������� ����������� �������.

### 6. **������� ���� ������**
- ��������� ��������� ���� ������ �� ������� ���������� ��� ������������� ������� � ������� ��.

---

## ������ ��� ��������� ������?

1. **������� ��������**: ��������� ������������� ��������� ����������, ��������� ������������ ������ ����������� � ������������� ���.
2. **��������**: ��������� ���������� ������� ������ ��������� ������������ ��������� ��� ��������� ������.
3. **�������� �������������**: ��������� �������������� ����������� ���������, ��� ������ �� ������� ��� ������������� ������ ������.
4. **����������**: ��������� ���� ��������� �������������� � �������������� ��� ������ � �������� ��������.


---

## �����������

��� ������ ��������� ���������� ��������� ���������� � ������:
- **Pillow (PIL)**: ��� ������ � �������������.
- **OpenCV (cv2)**: ��� ��������� ����������� � ������������� ������.
- **numpy**: ��� ������ � ���������.
- **face_data**: ���������������� ������ ��� ������ � ������� ���.
- **pinfacekirjasto**: ���������������� ������ ��� ����������� � ������������� ���.

---

## ������ �������������

1. ���������, ��� ��� ����������� �����������.
2. ������� ���� � �������� � ������������� (`pathinput`).
3. ��������� ������. ��������� ������������� ���������� ��� �����������, ������� ������ � ���� � ����������� ����� �� ������:
   - `/ok/` � ������� ������������ �����������.
   - `/double/` � ���������.
   - `/error/` � ����� � ��������.
   - `/empty/` � ����������� ��� ���.
   - `/many/` � ����������� � ����������� ������.

---

## �����

- **[������ ����������](mailto:kitaew@gmail.com)**

---

## �������������

- ������� ���������� **OpenCV** �� ��������������� ����������� ��� ������ � �������������.
- ������� ������������� **NumPy** �� ������ ���������� ��� ������ � ��������� ������.

---
# � �������� PIN

��������� ����������� ��� �������� **[PIN](https://pinspb.ru/)**, �������������� ���������� ��������-�����, ������� � 2006 ���� ������� ������������� �������������������� ������. PIN �������������� ���� ��� �������, ���������� � ������������� ��������, ��������� ���������������� ���� ���������� � ������.

## ������ PIN?
- **����������� ����������**: ���� PIN ��������� �� ��������� �����������, ��� ������������ ������� �������� ����� �� ��������� �����.
- **����������**: �������� ��������� ��������� ��������� ����������, ������������� ������������ ���� �������������.
- **����������� ������**: PIN ���������� ������ ������ IT-������� ��� ���� � �����, ������������ ����������� ��� �������, ��� � ������������� ��������.

---

## � ���������

��������� ������� ��� ��������� ������������� ������� PIN � ������� ��������� ������ � �������������. ��� �������� ���������� �������� � ���������������� ������������ � �������� �������� �����.

---

## ����������

PIN � ��� �� ������ ���������, ��� �������, ������� �������� ����� �������� ��������� ������ � ������� ����������� ����������. �� �������� ���, ��� �������� � ����� ������������� ���������!

## ������

- [������������ OpenCV](https://docs.opencv.org/)
- [������������ NumPy](https://numpy.org/doc/)
- [����������� �� GitHub](https://github.com/yourusername/yourrepository)