#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <iostream>
#include <optional>
#include <algorithm>

#define EPSILON 0.1f

template <typename T>
struct Vector2D
{
	T x;
	T y;
	
	template <typename K>
	void operator+=(const Vector2D<K>& v) { this->x += v.x; this->y += v.y; }

	template <typename K>
	void operator-=(const Vector2D<K>& v) { this->x -= v.x; this->y -= v.y; }

	template <typename K>
	void operator*=(const Vector2D<K>& v) { this->x *= v.x; this->y *= v.y; }

	template <typename K>
	void operator/=(const Vector2D<K>& v) { this->x /= v.x; this->y /= v.y; }

	template <typename K>
	void operator+=(const K& n) { this->x += n; this->y += n; }

	template <typename K>
	void operator-=(const K& n) { this->x -= n; this->y -= n; }

	template <typename K>
	void operator*=(const K& n) { this->x *= n; this->y *= n; }

	template <typename K>
	void operator/=(const K& n) { this->x /= n; this->y /= n; }

	template<typename K>
	bool operator==(const Vector2D<K>& v) { return this->x == v.x && this->y == v.y; }

	template<typename K>
	bool operator!=(const Vector2D<K>& v) { return this->x != v.x || this->y != v.y; }

	template <typename T, typename K>
	[[nodiscard]] static float Dot(const Vector2D<T>& a, const Vector2D<K>& b)
	{
		return a.x * b.x + a.y * b.y;
	}
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const Vector2D<T>& v)
{
	os << '(' << v.x << ", " << v.y << ")";
	return os;
}

template <typename T, typename K>
[[nodiscard]] Vector2D<T> operator+(const Vector2D<T>& a, const Vector2D<K>& b) { return Vector2D<T>{ a.x + b.x, a.y + b.y }; }

template <typename T, typename K>
[[nodiscard]] Vector2D<T> operator-(const Vector2D<T>& a, const Vector2D<K>& b) { return Vector2D<T>{ a.x - b.x, a.y - b.y }; }

template <typename T, typename K>
[[nodiscard]] Vector2D<T> operator*(const Vector2D<T>& a, const Vector2D<K>& b) { return Vector2D<T>{ a.x * b.x, a.y * b.y }; }

template <typename T, typename K>
[[nodiscard]] Vector2D<T> operator/(const Vector2D<T>& a, const Vector2D<K>& b) { return Vector2D<T>{ a.x / b.x, a.y / b.y }; }

template <typename T, typename K>
[[nodiscard]] Vector2D<T> operator+(const Vector2D<T>& v, const K& n) { return Vector2D<T>{ v.x + n, v.y + n }; }

template <typename T, typename K>
[[nodiscard]] Vector2D<T> operator-(const Vector2D<T>& v, const K& n) { return Vector2D<T>{ v.x - n, v.y - n }; }

template <typename T, typename K>
[[nodiscard]] Vector2D<T> operator*(const Vector2D<T>& v, const K& n) { return Vector2D<T>{ v.x * n, v.y * n }; }

template <typename T, typename K>
[[nodiscard]] Vector2D<T> operator/(const Vector2D<T>& v, const K& n) { return Vector2D<T>{ v.x / n, v.y / n }; }

template <typename T>
struct Vector3D : Vector2D<T>
{
	T z;

	template <typename K>
	void operator+=(const Vector3D<K>& v) { this->x += v.x; this->y += v.y; this->z += v.z; }

	template <typename K>
	void operator-=(const Vector3D<K>& v) { this->x -= v.x; this->y -= v.y; this->z -= v.z; }

	template <typename K>
	void operator*=(const Vector3D<K>& v) { this->x *= v.x; this->y *= v.y; this->z *= v.z; }

	template <typename K>
	void operator/=(const Vector3D<K>& v) { this->x /= v.x; this->y /= v.y; this->z /= v.z; }

	template <typename K>
	void operator+=(const K& n) { this->x += n; this->y += n; this->z += n; }

	template <typename K>
	void operator-=(const K& n) { this->x -= n; this->y -= n; this->z -= n; }

	template <typename K>
	void operator*=(const K& n) { this->x *= n; this->y *= n; this->z *= n; }

	template <typename K>
	void operator/=(const K& n) { this->x /= n; this->y /= n; this->z /= n; }

	template<typename K>
	bool operator==(const Vector3D<K>& v) { return this->x == v.x && this->y == v.y && this->z == v.z; }

	template<typename K>
	bool operator!=(const Vector3D<K>& v) { return this->x != v.x || this->y != v.y || this->z != v.z; }

	float Length()
	{
		return std::sqrt( x*x+y*y+z*z );
	}

	void Normalize()
	{
		float l = Length();
		operator/=(l);
	}

	template <typename T, typename K>
	[[nodiscard]] static float Dot(const Vector3D<T>& a, const Vector3D<K>& b)
	{
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}
};

template <typename T, typename K>
[[nodiscard]] Vector3D<T> operator+(const Vector3D<T>& a, const Vector3D<K>& b) { return Vector3D<T>{ static_cast<T>(a.x + b.x), static_cast<T>(a.y + b.y), static_cast<T>(a.z + b.z) }; }

template <typename T, typename K>
[[nodiscard]] Vector3D<T> operator-(const Vector3D<T>& a, const Vector3D<K>& b) { return Vector3D<T>{ static_cast<T>(a.x - b.x), static_cast<T>(a.y - b.y), static_cast<T>(a.z - b.z) }; }

template <typename T, typename K>
[[nodiscard]] Vector3D<T> operator*(const Vector3D<T>& a, const Vector3D<K>& b) { return Vector3D<T>{ static_cast<T>(a.x * b.x), static_cast<T>(a.y * b.y), static_cast<T>(a.z * b.z) }; }

template <typename T, typename K>
[[nodiscard]] Vector3D<T> operator/(const Vector3D<T>& a, const Vector3D<K>& b) { return Vector3D<T>{ static_cast<T>(a.x / b.x), static_cast<T>(a.y / b.y), static_cast<T>(a.z / b.z) }; }

template <typename T, typename K>
[[nodiscard]] Vector3D<T> operator+(const Vector3D<T>& v, const K& n) { return Vector3D<T>{ static_cast<T>(v.x + n), static_cast<T>(v.y + n), static_cast<T>(v.z + n) }; }

template <typename T, typename K>
[[nodiscard]] Vector3D<T> operator-(const Vector3D<T>& v, const K& n) { return Vector3D<T>{ static_cast<T>(v.x - n), static_cast<T>(v.y - n), static_cast<T>(v.z - n) }; }

template <typename T, typename K>
[[nodiscard]] Vector3D<T> operator*(const Vector3D<T>& v, const K& n) { return Vector3D<T>{ static_cast<T>(v.x * n), static_cast<T>(v.y * n), static_cast<T>(v.z * n) }; }

template <typename T, typename K>
[[nodiscard]] Vector3D<T> operator/(const Vector3D<T>& v, const K& n) { return Vector3D<T>{ static_cast<T>(v.x / n), static_cast<T>(v.y / n), static_cast<T>(v.z / n) }; }

template <typename T>
std::ostream& operator<<(std::ostream& os, const Vector3D<T>& v)
{
	os << '(' << v.x << ", " << v.y << ", " << v.z << ")";
	return os;
}

typedef Vector2D<float>    Vec2f;
typedef Vector2D<int8_t>   Vec2i8;
typedef Vector2D<int16_t>  Vec2i16;
typedef Vector2D<int32_t>  Vec2i32;
typedef Vector2D<int64_t>  Vec2i64;
typedef Vector2D<uint8_t>  Vec2u8;
typedef Vector2D<uint16_t> Vec2u16;
typedef Vector2D<uint32_t> Vec2u32;
typedef Vector2D<uint64_t> Vec2u64;

typedef Vector3D<float>    Vec3f;
typedef Vector3D<int8_t>   Vec3i8;
typedef Vector3D<int16_t>  Vec3i16;
typedef Vector3D<int32_t>  Vec3i32;
typedef Vector3D<int64_t>  Vec3i64;
typedef Vector3D<uint8_t>  Vec3u8;
typedef Vector3D<uint16_t> Vec3u16;
typedef Vector3D<uint32_t> Vec3u32;
typedef Vector3D<uint64_t> Vec3u64;

struct Color {
    uint8_t r;
    uint8_t g;
    uint8_t b;


	operator Vec3u16 () const noexcept {
		return Vec3u16{ r,g,b };
	}
};

class Image {
public:
    uint16_t width, height;

    Color* m_buff;

public:
    Image(const uint16_t width, const uint16_t height)
        : width(width), height(height)
    {
        this->m_buff = new Color[width * height];
        std::memset(this->m_buff, 0u, this->width * this->height * sizeof(Color));
    }

	void SetPixel(const uint16_t x, const uint16_t y, const Color& color)
	{
		//std::cout << (uint16_t)color.r << '\n';
		this->m_buff[y*width+x] = color;
	}

    void Save(const char* filename) {
		FILE* file = fopen(filename, "wb");

		const std::string header = "P6\n" + std::to_string(this->width) + " " + std::to_string(this->height) + " 255\n";

		fprintf(file, header.c_str());
		fwrite(this->m_buff, this->width * this->height, sizeof(Color), file);

		fclose(file);
    }

    ~Image()
    {
        delete this->m_buff;
    }
};

struct Ray {
	Vec3f origin;
	Vec3f direction;
};

struct Object;

struct Intersect {
	double d;
	Vec3f point;
	Object* object;
};

struct Object {
	Vec3f position;
	Color color;

	virtual Vec3f GetNormal(const Vec3f& p) const noexcept = 0;
	virtual bool Intersects(const Ray& ray) const noexcept = 0;
	virtual std::vector<Intersect> GetIntersects(const Ray& ray) const noexcept = 0;
};

struct Sphere : Object {
	float radius;

	Sphere() {}
	Sphere(Vec3f position, float radius, Color color) : radius(radius)
	{
		this->color = color;
		this->position = position;
	}

	virtual Vec3f GetNormal(const Vec3f& p) const noexcept override
	{
		Vec3f normal = (p-this->position)/ (float) this->radius;
		normal.Normalize();
		return normal;
	}

	virtual bool Intersects(const Ray& ray) const noexcept override
	{
		Vec3f oc = ray.origin-this->position;
		double b = 2*Vec3f::Dot(oc, ray.direction);
		double c = Vec3f::Dot(oc,oc)-this->radius*this->radius;
		double deltaSqrt = b*b-4*c;

		return (deltaSqrt >= 0.f);
	}

	virtual std::vector<Intersect> GetIntersects(const Ray& ray) const noexcept override
	{
		Vec3f L = this->position - ray.origin; 
        float tca = Vec3f::Dot(L, ray.direction); 
        float d2 = Vec3f::Dot(L, L) - tca * tca; 
        if (d2 > this->radius) return {}; 
        float thc = sqrt(this->radius - d2); 
        float t0 = tca - thc; 
        float t1 = tca + thc; 

		if (t0 > t1) std::swap(t0, t1); 
 
        if (t0 < 0) { 
            t0 = t1;
            if (t0 < 0) return { };
        } 
 
        float t = t0; 
 
		return std::vector<Intersect>{
			Intersect{t, ray.origin + ray.direction*t, (Object*)this}
		};
	}
};

struct Camera {
	Vec3f position {0.f, 0.f, 0.1f};

	Ray GenerateRay(const Vec2u16& pixel, const Vec2u16& renderSurfaceDims)
	{
		Vec3f nPixelWorldPosition = {
			2*(pixel.x + 1u) / (float)renderSurfaceDims.x - 1.f,
			2*(pixel.y + 1u) / (float)renderSurfaceDims.y - 1.f,
			1.f
		};

		nPixelWorldPosition.Normalize();

		return Ray { this->position, nPixelWorldPosition };
	}
};

struct Light {
	Vec3f position;
	Color color;
	float intensity;
};

struct Scene {
	Camera camera;
	std::vector<Object*> objects;
	std::vector<Light> lights;

	std::optional<Intersect> GetClosestRayIntersection(const Ray& ray)
	{
		Intersect closestIntersection;
		closestIntersection.object = nullptr;

		for (const Object* object : this->objects)
			for (const Intersect& intersect : object->GetIntersects(ray))
				if (closestIntersection.object == nullptr || intersect.d < closestIntersection.d)
					closestIntersection = intersect;

		if (closestIntersection.object == nullptr) return { };

		return closestIntersection;
	}

	void Draw(Image& renderSurface)
	{
		const Vec2u16 renderSurfaceDims = { renderSurface.width, renderSurface.height };

		Vec2u16 pixelPosition;
		for (pixelPosition.x = 0; pixelPosition.x < renderSurface.width; pixelPosition.x++) {
			for (pixelPosition.y = 0; pixelPosition.y < renderSurface.height; pixelPosition.y++) {
				const Ray cameraRay = camera.GenerateRay(pixelPosition, renderSurfaceDims);

				const std::optional<Intersect> cameraIntersection = this->GetClosestRayIntersection(cameraRay);
 
				if (cameraIntersection.has_value())
				{
					Vec3u16 compoundedColorVec{0u,0u,0u};

					for (const Light& light : this->lights)
					{
						Vec3f pointToLight = light.position - cameraIntersection.value().point;
						const float distanceToLight = pointToLight.Length();
						pointToLight.Normalize();

						const Vec3f normal = cameraIntersection.value().object->GetNormal(cameraIntersection.value().point);

						const Ray pointToLightRay = {
							cameraIntersection.value().point + normal * EPSILON,
							pointToLight
						};

						bool bIsInShadow = false;

						const std::optional<Intersect> lightIntersection = this->GetClosestRayIntersection(pointToLightRay);

						if (lightIntersection.has_value())
							if (lightIntersection.value().d < distanceToLight)
								bIsInShadow = true;

						if (!bIsInShadow)
						{
							float dotProduct = std::max(std::min(Vec3f::Dot(pointToLight, normal), 1.f), 0.f);
							float intensity  = dotProduct;

							compoundedColorVec.x += intensity * cameraIntersection.value().object->color.r;
							compoundedColorVec.y += intensity * cameraIntersection.value().object->color.g;
							compoundedColorVec.z += intensity * cameraIntersection.value().object->color.b;
						}
					}

					Color pixelColor;
					pixelColor.r = (compoundedColorVec.x <= 255u) ? compoundedColorVec.x : 255u;
					pixelColor.g = (compoundedColorVec.y <= 255u) ? compoundedColorVec.y : 255u;
					pixelColor.b = (compoundedColorVec.z <= 255u) ? compoundedColorVec.z : 255u;

					renderSurface.SetPixel(pixelPosition.x, pixelPosition.y, pixelColor);
				}
			}
		}
	}
};

int main()
{
    Image renderSurface(1000u, 1000u);
	Scene scene;

	scene.objects.push_back(new Sphere(Vec3f{-1,0,3.5f}, 1.f, Color{255,51,51}));
	scene.objects.push_back(new Sphere(Vec3f{+1,0,3.f}, 1.f, Color{51,51,255}));

	scene.lights.push_back(Light{Vec3f{3.f, 0.f, 0.5f}, Color{255,255,255}, 0.5f});

	scene.Draw(renderSurface);

	renderSurface.Save("frame.ppm");

    return 0;
}