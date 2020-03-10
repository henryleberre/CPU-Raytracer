#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <iostream>
#include <optional>
#include <algorithm>

#define EPSILON 0.01f
#define MAX_RECURSION_DEPTH 5

template <typename T>
struct Vector2D
{
	union {
		struct { T x; T y; };
		struct { T r; T g; };
	};
	
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
	union {
		struct { T z; };
		struct { T b; };
	};

	template <typename K>
	operator Vector3D<K>() const noexcept {
		return Vector3D<K>{
			static_cast<K>(this->x),
			static_cast<K>(this->y),
			static_cast<K>(this->z)
		};
	}

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

	template <typename T, typename K>
	[[nodiscard]] static Vector3D<T> Cross(const Vector3D<T>& a, const Vector3D<K>& b)
	{
		return Vector3D<T>{
			static_cast<T>(a.y * b.z - a.z * b.y),
			static_cast<T>(a.z * b.x - a.x * b.z),
			static_cast<T>(a.x * b.y - a.y * b.x)
		};
	}

	template <typename T, typename K>
	[[nodiscard]] static Vector3D<T> Reflect(const Vector3D<T>& in, const Vector3D<K>& normal)
	{
		return in - normal * 2 * Vec3f::Dot(in, normal);
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

typedef Vec3u8  Coloru8;
typedef Vec3u16 Coloru32;
typedef Vec3f   Colorf;

class Image {
public:
    uint16_t width, height;

    Coloru8* m_buff;

public:
    Image(const uint16_t width, const uint16_t height)
        : width(width), height(height)
    {
        this->m_buff = new Coloru8[width * height];
        std::memset(this->m_buff, 0u, this->width * this->height * sizeof(Coloru8));
    }

	void SetPixel(const uint16_t x, const uint16_t y, const Coloru8& color)
	{
		//std::cout << (uint16_t)color.r << '\n';
		this->m_buff[y*width+x] = color;
	}

    void Save(const char* filename) {
		FILE* file = fopen(filename, "wb");

		const std::string header = "P6\n" + std::to_string(this->width) + " " + std::to_string(this->height) + " 255\n";

		fprintf(file, header.c_str());
		fwrite(this->m_buff, this->width * this->height, sizeof(Coloru8), file);

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
	Coloru8 color;

	float reflectivity = 0.f;

	virtual Vec3f GetNormal(const Vec3f& p) const noexcept = 0;
	virtual std::optional<Intersect> GetIntersect(const Ray& ray) const noexcept = 0;

	virtual ~Object() {}
};

struct Plane : Object {
	Plane() {}

	Plane(const Vec3f& position, const Coloru8& color, const float reflectivity)
	{
		this->position = position;
		this->color = color;
		this->reflectivity = reflectivity;
	}

	~Plane() { }

	virtual Vec3f GetNormal(const Vec3f& p = Vec3f{}) const noexcept override
	{
		return {0.f, 1.f, 0.f};
	}

	virtual std::optional<Intersect> GetIntersect(const Ray& ray) const noexcept override
	{
		const float denom = Vec3f::Dot(GetNormal(), ray.direction);
		if (std::abs(denom) > 1e-6)
		{
			const Vec3f v = this->position - ray.origin;
			const float t = Vec3f::Dot(v, GetNormal()) / denom;

			if (t >= 0)
				return Intersect{t, ray.origin+ray.direction*t, (Object*)this};
		}

		return {};
	}
};

struct Sphere : Object {
	float radius;

	Sphere() {}
	Sphere(const Vec3f& position, const float radius, const Coloru8& color, const float reflectivity) : radius(radius)
	{
		this->color = color;
		this->position = position;
		this->reflectivity = reflectivity;
	}

	~Sphere() {}

	virtual Vec3f GetNormal(const Vec3f& p) const noexcept override
	{
		Vec3f normal = (p-this->position)/ (float) this->radius;
		normal.Normalize();
		return normal;
	}

	virtual std::optional<Intersect> GetIntersect(const Ray& ray) const noexcept override
	{
		Vec3f L = this->position - ray.origin; 
        float tca = Vec3f::Dot(L, ray.direction); 
        float d2 = Vec3f::Dot(L, L) - tca * tca; 

        if (d2 > this->radius)
			return {};
        
		float thc = sqrt(this->radius - d2); 
        float t0 = tca - thc; 
        float t1 = tca + thc; 

		if (t0 > t1) std::swap(t0, t1); 
 
        if (t0 < 0) { 
            t0 = t1;
            if (t0 < 0) return { };
        } 
 
        float t = t0; 
 
		return Intersect{t, ray.origin + ray.direction*t, (Object*)this};
	}
};

struct Camera {
	Vec3f position {0.f, 0.f, 0.1f};

	Ray GenerateRay(const Vec2u16& pixel, const Vec2u16& renderSurfaceDims)
	{
		const float aspectRatio = renderSurfaceDims.x/renderSurfaceDims.y;
		Vec3f nPixelWorldPosition = {
			2*(pixel.x + 1u) / (float)renderSurfaceDims.x - 1.f,
			2*(renderSurfaceDims.y - pixel.y + 1u) / (float)renderSurfaceDims.y - 1.f,
			1.f
		};

		nPixelWorldPosition -= this->position;
		nPixelWorldPosition.Normalize();

		return Ray { this->position, nPixelWorldPosition };
	}
};

struct Light {
	Vec3f   position;
	Coloru8 color;
	float   intensity;
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
		{
			const std::optional<Intersect>& intersect = object->GetIntersect(ray);

			if (intersect.has_value())
				if (closestIntersection.object == nullptr || intersect.value().d < closestIntersection.d)
					closestIntersection = intersect.value();
		}

		if (closestIntersection.object == nullptr) return { };

		return closestIntersection;
	}

	Coloru8 ComputeColor(const Ray& ray, const uint16_t currentDepth = 0u)
	{
		Coloru32 pixelColor{10u,10u,10u};
		const std::optional<Intersect> closestIntersectOptional = this->GetClosestRayIntersection(ray);

		if (!closestIntersectOptional.has_value())
			return pixelColor;

		const Intersect& intersection = closestIntersectOptional.value();
		const Object&    object = (*intersection.object);
		const Vec3f      normal = object.GetNormal(intersection.point);

		float totalDiffuseIntensity  = 0.f;
		float totalLightingIntensity = 0.f;
		for (const Light& light : this->lights)
		{
			Vec3f pointToLight = light.position - intersection.point;
			const float distanceToLight = pointToLight.Length();
			pointToLight.Normalize();

			const Ray pointToLightRay = {
				intersection.point + normal * EPSILON,
				pointToLight
			};

			const std::optional<Intersect> lightIntersectionOptional = this->GetClosestRayIntersection(pointToLightRay);
			
			bool isInShadow = false;
			if (lightIntersectionOptional.has_value())
				isInShadow = lightIntersectionOptional.value().d < distanceToLight;

			if (!isInShadow)
			{
				const float dotProduct = std::clamp(Vec3f::Dot(pointToLight, normal), 0.f, 1.f);
				
				const float lightingIntensity = light.intensity * dotProduct;

				totalLightingIntensity += lightingIntensity;
				totalDiffuseIntensity  += lightingIntensity * (1.f - object.reflectivity);
			}
		}

		totalDiffuseIntensity  = std::clamp(totalDiffuseIntensity,  0.f, 1.f);
		totalLightingIntensity = std::clamp(totalLightingIntensity, 0.f, 1.f);

		pixelColor += object.color * totalDiffuseIntensity;

		if (object.reflectivity > EPSILON)
		{
			const Ray reflectionRay = {
				intersection.point + normal * EPSILON,
				Vec3f::Reflect(ray.direction, normal)
			};

			if (MAX_RECURSION_DEPTH != currentDepth)
				pixelColor += ComputeColor(reflectionRay, currentDepth + 1u) * totalLightingIntensity * object.reflectivity;
		}

		pixelColor.r = std::clamp(pixelColor.r, (uint16_t)0u, (uint16_t)255u);
		pixelColor.g = std::clamp(pixelColor.g, (uint16_t)0u, (uint16_t)255u);
		pixelColor.b = std::clamp(pixelColor.b, (uint16_t)0u, (uint16_t)255u);

		return pixelColor;
	}

	void Draw(Image& renderSurface)
	{
		const Vec2u16 renderSurfaceDims = { renderSurface.width, renderSurface.height };

		Vec2u16 pixelPosition;
		for (pixelPosition.x = 0; pixelPosition.x < renderSurface.width; pixelPosition.x++) {
			for (pixelPosition.y = 0; pixelPosition.y < renderSurface.height; pixelPosition.y++) {
				const Ray cameraRay = camera.GenerateRay(pixelPosition, renderSurfaceDims);

				renderSurface.SetPixel(pixelPosition.x, pixelPosition.y, ComputeColor(cameraRay));
			}
		}
	}
};

int main()
{
    Image renderSurface(1000, 1000);
	Scene scene;

	scene.objects.push_back(new Sphere(Vec3f{-3.0f,  1.0f, 3.0f}, 1.0f, Coloru8{0,   255, 0  }, 0.f));
	scene.objects.push_back(new Sphere(Vec3f{-1.0f,  0.4f, 5.0f}, 1.0f, Coloru8{255, 255, 255}, 1.0f));
	scene.objects.push_back(new Sphere(Vec3f{+1,     0.4f, 3.0f}, 1.0f, Coloru8{51,  51,  255}, 0.0f));
	scene.objects.push_back(new Sphere(Vec3f{0.75f, -1.0f, 2.5f}, 1.0f, Coloru8{0,   255, 255}, 0.0f));
	scene.objects.push_back(new Sphere(Vec3f{-2.0f,  2.0f, 6.0f}, 1.0f, Coloru8{255, 0,   0  }, 0.0f));

	scene.objects.push_back(new Plane(Vec3f{-1.f, -2.f, 0.f}, Coloru8{255, 255, 255}, 0.f));

	//scene.lights.push_back(Light{Vec3f{-3.f, 0.f, 0.f},  Coloru8{255,255,255}, 0.5f});
	//scene.lights.push_back(Light{Vec3f{+3.f, 0.f, 0.f},  Coloru8{255,255,255}, 0.5f});
	//scene.lights.push_back(Light{Vec3f{-3.f, 0.f, 10.f}, Coloru8{255,255,255}, 0.5f});
	//scene.lights.push_back(Light{Vec3f{+3.f, 0.f, 10.f}, Coloru8{255,255,255}, 0.5f});
	//scene.lights.push_back(Light{Vec3f{+3.f, 4.f, -1.f},  Coloru8{255,255,255}, 1.f});
	scene.lights.push_back(Light{Vec3f{-1.f, 1.5f, -2.f}, Coloru8{255,255,255}, 1.f});

	scene.Draw(renderSurface);

	renderSurface.Save("frame.ppm");

    return 0;
}
