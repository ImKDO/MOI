import numpy as np

# --- КЛАССЫ ДЛЯ СТРУКТУРИРОВАНИЯ ДАННЫХ ---
class LightSource:
    def __init__(self, id, I0, O, PL):
        self.id = id
        self.I0 = np.array(I0)   # «Цветная» сила излучения I0 (RGB)
        self.O = np.array(O) / np.linalg.norm(O)  # Ось источника (нормализованная)
        self.PL = np.array(PL)   # Положение источника PL

class Material:
    def __init__(self, K_RGB, kd, ks, ke):
        self.K_RGB = np.array(K_RGB) # Цвет поверхности
        self.kd = kd                 # Коэф. диффузного отражения
        self.ks = ks                 # Коэф. зеркального отражения
        self.ke = ke                 # Ширина блика

# --- ОСНОВНЫЕ ФУНКЦИИ РАСЧЕТА ---
def calculate_lab_1():
    # 1. ВХОДНЫЕ ДАННЫЕ
    
    # Вершины треугольника (Изменены координаты P1 и P2, чтобы нормаль смотрела в +Z)
    P0 = np.array([0.0, 0.0, 0.0])
    P1 = np.array([0.0, 10.0, 0.0]) # Было [10, 0, 0]
    P2 = np.array([10.0, 0.0, 0.0]) # Было [0, 10, 0]

    # Источники света 
    # Свет 1 направлен ровно вниз (0, 0, -1)
    # Свет 2 направлен под углом (1, 1, -1)
    lights = [
        LightSource(1, I0=[200, 200, 200], O=[0, 0, -1], PL=[5, 5, 10]),
        LightSource(2, I0=[100, 150, 200], O=[1, 1, -1], PL=[-2, 10, 8])
    ]

    # Параметры наблюдения и материала
    V = np.array([0.0, 0.0, 1.0]) # Направление наблюдения (камера смотрит сверху вниз)
    V = V / np.linalg.norm(V)
    
    mat = Material(K_RGB=[0.8, 0.8, 0.8], kd=0.7, ks=0.5, ke=20)

    # Сетка локальных координат (x, y) для таблиц в отчете
    x_steps = [1.0, 2.5, 4.0, 5.5, 7.0]
    y_steps = [1.0, 2.5, 4.0, 5.5, 7.0]

    # 4. Вычисление вектора нормали плоскости (Формула 4)
    edge1 = P1 - P0
    edge2 = P2 - P0
    # Строго по методичке: (P2-P0) x (P1-P0)
    N_vec = np.cross(edge2, edge1) 
    N = N_vec / np.linalg.norm(N_vec)

    # Подготовка массивов для хранения результатов таблиц
    table_E1 = [] # Освещенность от 1-го источника
    table_E2 = [] # Освещенность от 2-го источника
    table_L = []  # Итоговая яркость

    print(f"--- ПАРАМЕТРЫ СЦЕНЫ ---")
    print(f"Нормаль плоскости N: {N}")
    print(f"-----------------------\n")

    for y in y_steps:
        row_E1, row_E2, row_L = [], [], []
        for x in x_steps:
            # Ограничение "треугольником", чтобы видеть границы фигуры в таблице
            if (x + y) > 10.0:
                row_E1.append([0.0, 0.0, 0.0])
                row_E2.append([0.0, 0.0, 0.0])
                row_L.append([0.0, 0.0, 0.0])
                continue

            # 3. Перевод локальных координат в глобальные (Формула 3)
            e1_n = edge1 / np.linalg.norm(edge1)
            e2_n = edge2 / np.linalg.norm(edge2)
            PT = P0 + (e1_n * x + e2_n * y)

            point_E = [] # Для хранения E от каждого источника отдельно

            for light in lights:
                # 5. Вектор от точки до источника света
                s_vec = light.PL - PT
                R = np.linalg.norm(s_vec)
                s_dir = s_vec / R # Нормализованное направление НА свет (от поверхности)

                # Направление ОТ света К точке (для диаграммы излучения)
                dir_light_to_pt = -s_dir
                
                # cos(theta) - угол между осью источника и направлением на точку (Формула 1)
                cos_theta = np.dot(dir_light_to_pt, light.O)
                cos_theta = max(0, cos_theta) # Если < 0, источник светит в другую сторону

                # Цветная сила излучения I (Формула 1)
                I_color = light.I0 * cos_theta

                # cos(alpha) - угол падения света (Формула 2)
                cos_alpha = np.dot(s_dir, N)
                cos_alpha = max(0, cos_alpha) # Отсекаем свет, идущий из-под поверхности

                # Освещенность E (Формула 2)
                E = (I_color * cos_alpha) / (R**2)
                point_E.append(E)

            # 6,7,8. Расчет BRDF и итоговой яркости
            total_L = np.zeros(3)
            for i, light in enumerate(lights):
                s_dir = (light.PL - PT) / np.linalg.norm(light.PL - PT)
                
                # Средний вектор h (Формула 8)
                h = (V + s_dir) / np.linalg.norm(V + s_dir)

                # BRDF f (Формула 7)
                dot_hN = max(0, np.dot(h, N))
                f_brdf = mat.K_RGB * (mat.kd + mat.ks * (dot_hN ** mat.ke))

                # Суммируем вклад источника в яркость
                # Освещенность E_i модулируется на функцию BRDF
                total_L += point_E[i] * f_brdf

            # Деление на пи согласно модели Ламберта/Блинна-Фонга (Формула 6)
            total_L = total_L / np.pi

            row_E1.append(point_E[0])
            row_E2.append(point_E[1])
            row_L.append(total_L)

        table_E1.append(row_E1)
        table_E2.append(row_E2)
        table_L.append(row_L)

    # --- ВЫВОД РЕЗУЛЬТАТОВ В ВИДЕ ТАБЛИЦ ---
    def format_rgb(arr):
        """Хелпер для красивого вывода векторов в строку таблицы"""
        return f"[{arr[0]:4.1f}, {arr[1]:4.1f}, {arr[2]:4.1f}]"

    def print_table(title, data, x_headers, y_headers):
        print(f"{title}")
        header = "y \ x | " + " | ".join(f"{x:^18}" for x in x_headers)
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        for i, y_val in enumerate(y_headers):
            row_str = f"{y_val:<5} | " + " | ".join(f"{format_rgb(val):^18}" for val in data[i])
            print(row_str)
        print()

    print_table("Таблица 1: Освещенность E1 (от 1-го источника)", table_E1, x_steps, y_steps)
    print_table("Таблица 2: Освещенность E2 (от 2-го источника)", table_E2, x_steps, y_steps)
    print_table("Таблица 3: Итоговая яркость L(RGB) на поверхности", table_L, x_steps, y_steps)

if __name__ == "__main__":
    calculate_lab_1()