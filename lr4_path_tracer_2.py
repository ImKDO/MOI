import numpy as np

EPS = 1e-8


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < EPS:
        return v.copy()
    return v / n


def fresnel_schlick(cos_theta: float, f0: np.ndarray) -> np.ndarray:
    return f0 + (1.0 - f0) * ((1.0 - np.clip(cos_theta, 0.0, 1.0)) ** 5)


def distribution_ggx(n: np.ndarray, h: np.ndarray, roughness: float) -> float:
    a = max(roughness * roughness, 0.02)
    a2 = a * a
    ndoth = max(float(np.dot(n, h)), 0.0)
    ndoth2 = ndoth * ndoth
    denom = ndoth2 * (a2 - 1.0) + 1.0
    return a2 / max(np.pi * denom * denom, EPS)


def geometry_schlick_ggx(ndotx: float, roughness: float) -> float:
    r = max(roughness, 0.02) + 1.0
    k = (r * r) / 8.0
    return ndotx / max(ndotx * (1.0 - k) + k, EPS)


def geometry_smith(n: np.ndarray, v: np.ndarray, l: np.ndarray, roughness: float) -> float:
    ndotv = max(float(np.dot(n, v)), 0.0)
    ndotl = max(float(np.dot(n, l)), 0.0)
    return geometry_schlick_ggx(ndotv, roughness) * geometry_schlick_ggx(ndotl, roughness)


def cook_torrance_specular(n: np.ndarray, v: np.ndarray, l: np.ndarray, roughness: float, f0: np.ndarray) -> np.ndarray:
    h = normalize(v + l)
    ndotv = max(float(np.dot(n, v)), 0.0)
    ndotl = max(float(np.dot(n, l)), 0.0)
    vdoth = max(float(np.dot(v, h)), 0.0)

    if ndotv <= 0.0 or ndotl <= 0.0:
        return np.zeros(3)

    d = distribution_ggx(n, h, roughness)
    g = geometry_smith(n, v, l, roughness)
    f = fresnel_schlick(vdoth, f0)
    return (d * g * f) / max(4.0 * ndotv * ndotl, EPS)


def blinn_phong_specular(n: np.ndarray, v: np.ndarray, l: np.ndarray, shininess: float = 64.0) -> np.ndarray:
    h = normalize(v + l)
    ndoth = max(float(np.dot(n, h)), 0.0)
    return np.array([ndoth ** shininess] * 3)


def cosine_weighted_hemisphere_sample(normal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    r1 = rng.random()
    r2 = rng.random()
    phi = 2.0 * np.pi * r1
    x = np.cos(phi) * np.sqrt(r2)
    y = np.sin(phi) * np.sqrt(r2)
    z = np.sqrt(max(1.0 - r2, 0.0))

    normal = normalize(normal)
    tangent = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    tangent = normalize(np.cross(tangent, normal))
    bitangent = np.cross(normal, tangent)
    world = x * tangent + y * bitangent + z * normal
    return normalize(world)


def sample_next_direction(normal: np.ndarray, view_dir: np.ndarray, previous_event: str, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    n = normalize(normal)
    v = normalize(view_dir)

    # Исправление TODO(1):
    # Для диффузного предыдущего события продолжаем трассировку
    # по косинусно-взвешенной полусфере, а не пытаемся использовать
    # только зеркальный сценарий.
    if previous_event == "diffuse":
        return cosine_weighted_hemisphere_sample(n, rng)

    # Для "specular" и "camera" сохраняем отражение как раньше.
    return normalize(2.0 * np.dot(n, v) * n - v)


def shade_point(
    normal: np.ndarray,
    view_dir: np.ndarray,
    light_dir: np.ndarray,
    albedo: np.ndarray,
    roughness: float,
    metallic: float,
    method: str = "blinn_phong",
) -> np.ndarray:
    n = normalize(normal)
    v = normalize(view_dir)
    l = normalize(light_dir)

    ndotl = max(float(np.dot(n, l)), 0.0)
    if ndotl <= 0.0:
        return np.zeros(3)

    albedo = np.clip(albedo, 0.0, 1.0)
    metallic = float(np.clip(metallic, 0.0, 1.0))
    f0 = (1.0 - metallic) * np.array([0.04, 0.04, 0.04]) + metallic * albedo

    diffuse = (1.0 - metallic) * albedo / np.pi

    # TODO(2): второй метод BRDF: Cook-Torrance
    if method == "cook_torrance":
        specular = cook_torrance_specular(n, v, l, roughness, f0)
    elif method == "blinn_phong":
        specular = blinn_phong_specular(n, v, l)
    else:
        raise ValueError("Unknown shading method. Use 'blinn_phong' or 'cook_torrance'.")

    return (diffuse + specular) * ndotl
