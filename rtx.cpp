#define __WEISS__DISABLE_SIMD

#include <thread>
#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <iostream>
#include <optional>
#include <algorithm>

#include "Weiss-Math/WSMath.h"

using namespace WSMath;

#define EPSILON 0.01f
#define THREAD_MAX 5
#define MAX_RECURSION_DEPTH 10

class Image {
public:
    uint16_t width, height;

    RawVectorComponents<uint8_t, 3u>* m_buff;

public:
    Image(const uint16_t width, const uint16_t height)
        : width(width), height(height)
    {
        this->m_buff = new RawVectorComponents<uint8_t, 3u>[width * height];
        std::memset(this->m_buff, 0u, this->width * this->height * sizeof(RawVectorComponents<uint8_t, 3u>));
    }

	void SetPixel(const uint16_t x, const uint16_t y, const Coloru8& color)
	{
		this->m_buff[y*width+x] = color;
	}

    void Save(const char* filename) {
		FILE* file = fopen(filename, "wb");

		const std::string header = "P6\n" + std::to_string(this->width) + " " + std::to_string(this->height) + " 255\n";

		fprintf(file, header.c_str());
		fwrite(this->m_buff, this->width * this->height, sizeof(RawVectorComponents<uint8_t, 3u>), file);

		fclose(file);
    }

    ~Image()
    {
        delete this->m_buff;
    }
};

struct Ray {
	Vecf32 origin;
	Vecf32 direction;
};

struct Object;

struct Intersect {
	double d;
	Vecf32 point;
	Object* object;
};

struct Object {
	Vecf32 position;
	Colorf32 color;

	float reflectivity = 0.f;

	virtual Vecf32 GetNormal(const Vecf32& p) const noexcept = 0;
	virtual std::optional<Intersect> GetIntersect(const Ray& ray) const noexcept = 0;

	virtual ~Object() {}
};

struct Plane : Object {
	Vecf32 normal;
	
	Plane() {}

	Plane(const Vecf32& position, const Colorf32& color, const Vecf32& normal, const float reflectivity)
		: normal(normal)
	{
		this->position = position;
		this->color = color;
		this->reflectivity = reflectivity;
	}

	~Plane() { }

	virtual Vecf32 GetNormal(const Vecf32& p = Vecf32{}) const noexcept override
	{
		return this->normal;
	}

	virtual std::optional<Intersect> GetIntersect(const Ray& ray) const noexcept override
	{
		const float denom = Vector<>::DotProduct(GetNormal(), ray.direction);
		if (std::abs(denom) >= 1e-6)
		{
			const Vecf32 v = this->position - ray.origin;
			const float t = Vector<>::DotProduct(v, GetNormal()) / denom;

			if (t >= 0)
				return Intersect{t, ray.origin+ray.direction*t, (Object*)this};
		}

		return {};
	}
};

struct Sphere : Object {
	float radius;

	Sphere() {}
	Sphere(const Vecf32& position, const float radius, const Colorf32& color, const float reflectivity) : radius(radius)
	{
		this->color = color;
		this->position = position;
		this->reflectivity = reflectivity;
	}

	~Sphere() {}

	virtual Vecf32 GetNormal(const Vecf32& p) const noexcept override
	{
		return Vector<>::Normalized((p-this->position) / (float) this->radius);
	}

	virtual std::optional<Intersect> GetIntersect(const Ray& ray) const noexcept override
	{
		Vecf32 L = this->position - ray.origin; 

        float tca = Vector<>::DotProduct(L, ray.direction); 
        float d2 = Vector<>::DotProduct(L, L) - tca * tca; 

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
	Vecf32 position {0.f, 0.f, 0.1f};

	Ray GenerateRay(const Vecu16& pixel, const Vecu16& renderSurfaceDims)
	{
		const float aspectRatio = renderSurfaceDims.x/renderSurfaceDims.y;
		Vecf32 nPixelWorldPosition = {
			2*(pixel.x + 1u) / (float)renderSurfaceDims.x - 1.f,
			2*(renderSurfaceDims.y - pixel.y + 1u) / (float)renderSurfaceDims.y - 1.f,
			1.f
		};

		nPixelWorldPosition -= this->position;
		nPixelWorldPosition = Vector<>::Normalized(nPixelWorldPosition);

		return Ray { this->position, nPixelWorldPosition };
	}
};

struct Light {
	Vecf32   position;
	Colorf32 color;
	float    intensity;
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

	Colorf32 ComputeColor(const Ray& ray, const uint16_t currentDepth = 0u)
	{
		Colorf32 pixelColorf{0.05f,0.05f,0.05f};
		const std::optional<Intersect> closestIntersectOptional = this->GetClosestRayIntersection(ray);

		if (!closestIntersectOptional.has_value())
			return pixelColorf;

		const Intersect& intersection = closestIntersectOptional.value();
		const Object&    object = (*intersection.object);
		const Vecf32     normal = object.GetNormal(intersection.point);
		const Vecf32     pointToCamera = Vector<>::Normalized(camera.position - intersection.point);

		Colorf32 totalDiffuseColorf  = {0.f, 0.f, 0.f};
		Colorf32 totalSpecularColorf = {0.f, 0.f, 0.f};

		float totalLightingIntensity = 0.f;

		for (const Light& light : this->lights)
		{
			Vecf32 pointToLight = light.position - intersection.point;
			const float distanceToLight = pointToLight.GetLength();
			pointToLight = Vector<>::Normalized(pointToLight);

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
				// Diffuse
				const float diffuseDotProduct = std::clamp(Vector<>::DotProduct(pointToLight, normal), 0.f, 1.f);
				const float diffuseIntensity  = light.intensity * diffuseDotProduct;

				totalDiffuseColorf += light.color * diffuseIntensity * (1.f - object.reflectivity); // TODO:: CHANGE TO LIGHT COLOR

				// Specular
				float specularIntensity = 0.f;
				if (object.reflectivity > EPSILON)
				{
					const Vecf32 reflectedLight     = Vector<>::GetReflected(ray.direction, normal);
					const float  specularDotProduct = std::max(Vector<>::DotProduct(pointToCamera, reflectedLight), 0.f);
					specularIntensity = std::pow(specularDotProduct, 2.f);

					totalSpecularColorf += light.color * specularIntensity * object.reflectivity;
				}
				
				// Total
				totalLightingIntensity += diffuseIntensity + specularIntensity;
			}
		}

		totalLightingIntensity = std::clamp(totalLightingIntensity, 0.f, 1.f);

		pixelColorf += object.color * totalDiffuseColorf + totalSpecularColorf;

		if (object.reflectivity > EPSILON)
		{
			const Vecf32 cameraReflectionVector = Vector<>::GetReflected(ray.direction, normal);

			const Ray reflectionRay = {
				intersection.point + normal * EPSILON,
				cameraReflectionVector
			};

			if (MAX_RECURSION_DEPTH != currentDepth)
				pixelColorf += ComputeColor(reflectionRay, currentDepth + 1u) * totalLightingIntensity * object.reflectivity * 0.8f;
		}

		pixelColorf.r = std::clamp(pixelColorf.r, 0.f, 1.f);
		pixelColorf.g = std::clamp(pixelColorf.g, 0.f, 1.f);
		pixelColorf.b = std::clamp(pixelColorf.b, 0.f, 1.f);

		return pixelColorf;
	}

	void Draw(Image& renderSurface)
	{
		const Vecu16 renderSurfaceDims = { renderSurface.width, renderSurface.height };

		Vecu16 pixelPosition;
		for (pixelPosition.x = 0; pixelPosition.x < renderSurface.width; pixelPosition.x++) {
			for (pixelPosition.y = 0; pixelPosition.y < renderSurface.height; pixelPosition.y++) {
				const Ray cameraRay = camera.GenerateRay(pixelPosition, renderSurfaceDims);

				renderSurface.SetPixel(pixelPosition.x, pixelPosition.y, Coloru8(ComputeColor(cameraRay) * 255u));
			}
		}
	}

	~Scene()
	{
		for (Object* obj : this->objects)
			std::free(obj);
	}
};

int main()
{
    Image renderSurface(1000, 1000);
	Scene scene;

	// Ground
	scene.objects.push_back(new Plane(Vecf32{0.f, -1.5f, 0.f}, Colorf32{1.f, 1.f, 1.f}, Vecf32{0.f, 1.f, 0.f}, 0.0f));
	scene.objects.push_back(new Sphere(Vecf32{-2.f, 0.f, 5.f}, 1.f, Colorf32{1.f, 0.f, 0.f}, 1.f));
	scene.objects.push_back(new Sphere(Vecf32{2.f, 0.f, 5.f},  1.f, Colorf32{1.f, 0.f, 0.f}, 1.f));

	scene.lights.push_back(Light{Vecf32{0.f, 4.f, 0.f}, Colorf32{1.F,127/255.f,80/255.f}, 1.0f});
	scene.lights.push_back(Light{Vecf32{0.f, 0.f, 0.f}, Colorf32{1.F, 1.f, 1.F}, 1.0f});

	scene.Draw(renderSurface);

	renderSurface.Save("frame.ppm");

    return 0;
}
