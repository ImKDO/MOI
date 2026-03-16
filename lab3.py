import numpy as np
import matplotlib.pyplot as plt

N_POINTS = 100000

def create_local_basis(N):
    """Создает ортонормированный базис (U, V, N) вокруг вектора N."""
    N = N / np.linalg.norm(N)
    # Ищем вектор, не коллинеарный N
    A = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(N, A)) > 0.9: 
        A = np.array([0.0, 1.0, 0.0])
    U = np.cross(A, N)
    U = U / np.linalg.norm(U)
    V = np.cross(N, U)
    return U, V, N

def task1_triangle():
    print("--- Задача 1: Треугольник ---")
    V1 = np.array([0, 0, 0])
    V2 = np.array([10, 0, 0])
    V3 = np.array([5, 10, 5])

    r1 = np.random.rand(N_POINTS)
    r2 = np.random.rand(N_POINTS)
    
    # Отражение для гарантии нахождения в треугольнике
    mask = r1 + r2 > 1
    r1[mask] = 1 - r1[mask]
    r2[mask] = 1 - r2[mask]

    # Генерация точек
    pts = V1 + r1[:, None] * (V2 - V1) + r2[:, None] * (V3 - V1)

    # ДОКАЗАТЕЛЬСТВО:
    # 1. Проверяем барицентрические координаты (сумма < 1, u>0, v>0)
    assert np.all(r1 >= 0) and np.all(r2 >= 0) and np.all(r1 + r2 <= 1.00001), "Точки вне треугольника!"
    print(f"Сгенерировано {len(pts)} точек. Все лежат внутри треугольника (доказано барицентрикой).")

    # Графическое доказательство
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(pts[:2000, 0], pts[:2000, 1], pts[:2000, 2], s=1, alpha=0.5)
    ax1.set_title('3D Вид треугольника (первые 2000 точек)')
    
    ax2 = fig.add_subplot(122)
    ax2.hist2d(r1, r2, bins=50, cmap='Blues')
    ax2.set_title('Плотность вероятности (2D гистограмма)')
    plt.show()

def task2_circle():
    print("\n--- Задача 2: Круг ---")
    C = np.array([2, 2, 2])
    N = np.array([1, 1, 1]) # Произвольная нормаль
    Rc = 5.0
    
    U, V, N_norm = create_local_basis(N)

    r_rand = np.random.rand(N_POINTS)
    phi_rand = np.random.rand(N_POINTS) * 2 * np.pi

    # Радиус берется как корень, чтобы площадь распределялась равномерно!
    r = Rc * np.sqrt(r_rand) 
    
    x_local = r * np.cos(phi_rand)
    y_local = r * np.sin(phi_rand)

    pts = C + x_local[:, None] * U + y_local[:, None] * V

    # ДОКАЗАТЕЛЬСТВО:
    # 1. Расстояние от центра не превышает Rc
    distances = np.linalg.norm(pts - C, axis=1)
    assert np.all(distances <= Rc + 1e-6), "Есть точки вне радиуса!"
    
    # 2. Точки лежат в плоскости (скалярное произведение вектора на нормаль == 0)
    dot_products = np.abs(np.dot(pts - C, N_norm))
    assert np.max(dot_products) < 1e-10, "Точки не лежат в плоскости круга!"
    
    print(f"Сгенерировано {len(pts)} точек. Все лежат в плоскости и внутри радиуса R={Rc}.")

    # Графическое доказательство равномерности площади (гистограмма квадратов радиусов должна быть плоской)
    plt.figure(figsize=(5, 4))
    plt.hist(r**2, bins=50, density=True, color='skyblue', edgecolor='black')
    plt.title('Доказательство равномерности по площади (r^2)')
    plt.xlabel('r^2')
    plt.show()

def task3_sphere():
    print("\n--- Задача 3: Равномерная сфера ---")
    phi = np.random.rand(N_POINTS) * 2 * np.pi
    # Архимедово свойство: z равномерно от -1 до 1
    z = np.random.rand(N_POINTS) * 2 - 1 
    
    r_xy = np.sqrt(1 - z**2)
    x = r_xy * np.cos(phi)
    y = r_xy * np.sin(phi)
    
    pts = np.column_stack((x, y, z))

    # ДОКАЗАТЕЛЬСТВО:
    # 1. Длина каждого вектора = 1
    norms = np.linalg.norm(pts, axis=1)
    assert np.all(np.abs(norms - 1.0) < 1e-10), "Длины векторов не равны 1!"
    print(f"Сгенерировано {len(pts)} направлений. Все лежат на единичной сфере.")

    # Графическое доказательство (гистограмма Z)
    plt.figure(figsize=(5, 4))
    plt.hist(z, bins=50, density=True, color='lightgreen', edgecolor='black')
    plt.title('Доказательство равномерности: гистограмма Z \n(должна быть плоской от -1 до 1)')
    plt.xlabel('Координата Z (или cos(theta))')
    plt.show()

def task4_cosine_hemisphere():
    print("\n--- Задача 4: Косинусная полусфера ---")
    N = np.array([0, 0, 1]) # Относительно оси Z для простоты проверки, но работает для любой
    U, V, N_norm = create_local_basis(N)

    r1 = np.random.rand(N_POINTS)
    r2 = np.random.rand(N_POINTS)

    phi = 2 * np.pi * r1
    
    # Для косинусного распределения: cos(theta) = sqrt(r2)
    # Метод генерации Малли (Malley's method)
    z_local = np.sqrt(r2) 
    r_local = np.sqrt(1 - r2) # sin(theta)
    
    x_local = r_local * np.cos(phi)
    y_local = r_local * np.sin(phi)

    pts = x_local[:, None] * U + y_local[:, None] * V + z_local[:, None] * N_norm

    # ДОКАЗАТЕЛЬСТВО:
    # 1. Длина = 1
    norms = np.linalg.norm(pts, axis=1)
    assert np.all(np.abs(norms - 1.0) < 1e-10), "Длины векторов не равны 1!"
    
    # 2. Лежат в полусфере нормали N
    cos_theta = np.dot(pts, N_norm)
    assert np.all(cos_theta >= 0), "Точки попали за пределы полусферы!"
    
    print(f"Сгенерировано {len(pts)} направлений. Все лежат на единичной полусфере.")

    # Графическое доказательство:
    # Если распределение косинусное p(theta) ~ cos(theta), 
    # то плотность вероятности для z = cos(theta) равна f(z) = 2z (линейно возрастает)
    plt.figure(figsize=(5, 4))
    count, bins, ignored = plt.hist(cos_theta, bins=50, density=True, color='salmon', edgecolor='black')
    plt.plot(bins, 2*bins, linewidth=2, color='red', label='Теоретическая плотность f(z) = 2z')
    plt.title('Доказательство косинусного распределения \n гистограмма cos(theta)')
    plt.xlabel('cos(theta)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    task1_triangle()
    task2_circle()
    task3_sphere()
    task4_cosine_hemisphere()