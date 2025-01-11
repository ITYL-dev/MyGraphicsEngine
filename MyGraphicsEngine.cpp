#define _CRT_SECURE_NO_WARNINGS 1
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define M_PI 3.14159265358
#define MAX_LIGHT_INTENSITY 2e8
#define GAMMA 2.2
#define EPSILON 1e-6

#include <iostream>
#include <limits>

static inline double sqr(double x) { return x * x; }

unsigned char color_correction(double num) {

    num = pow(num, 1/GAMMA); // correction gamma
    if (num > 255) return 255; // clamping supérieur
    else if (num < 0) return 0; // clamping clamping inférieur
    else return static_cast<unsigned char>(num); // conversion
};

class Vector {
public:
    explicit Vector(double x = 0, double y = 0, double z = 0) {
        coord[0] = x;
        coord[1] = y;
        coord[2] = z;
    }
    double& operator[](int i) { return coord[i]; }
    double operator[](int i) const { return coord[i]; }

    Vector& operator+=(const Vector& v) {
        coord[0] += v[0];
        coord[1] += v[1];
        coord[2] += v[2];
        return *this;
    }

    double norm2() const {
        return sqr(coord[0]) + sqr(coord[1]) + sqr(coord[2]);
    }

    void normalize() {
        double norm{ sqrt(this->norm2()) };
        coord[0] /= norm;
        coord[1] /= norm;
        coord[2] /= norm;
    };

    double coord[3];
};

Vector operator+(const Vector& a, const Vector& b) {
    return Vector(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}
Vector operator-(const Vector& a, const Vector& b) {
    return Vector(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}
Vector operator*(const Vector& a, double b) {
    return Vector(a[0]*b, a[1]*b, a[2]*b);
}
Vector operator*(double a, const Vector& b) {
    return Vector(a*b[0], a*b[1], a*b[2]);
}

double dot(const Vector& a, const Vector& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

class Ray {
public:
    
    Ray(const Vector& origin, const Vector& direction): origin(origin), direction(direction) {};

    Vector origin, direction;
};

class Sphere {
public:

    Sphere(const Vector& center, double radius, const Vector& albedo=Vector(1,1,1)) : center(center), radius(radius), albedo(albedo) {};

    bool intersect(const Ray& ray, Vector& intersection_point, Vector& intersection_normal, double& t) {
        double a{ 1 };
        double b = 2 * dot(ray.direction, ray.origin - center);
        double c = (ray.origin - center).norm2() - sqr(radius);

        double delta{ sqr(b) - 4 * a * c };

        if (delta < 0) return false; // pas d'intersection

        double sqrt_delta{ sqrt(delta) };
        double t1{ (-b - sqrt_delta) / (2 * a) };
        double t2{ (-b + sqrt_delta) / (2 * a) };

        if (t2 < 0) return false; // intersection pas du bon côté de la caméra

        if (t1 >= 0) {
            t = t1;
        } else {
            t = t2; // la caméra est dans la sphère
        }

        intersection_point = ray.origin + t * ray.direction; // point d'intersection
        intersection_normal = intersection_point - center; // vecteur normal à la sphère en intersection_point
        intersection_normal.normalize();

        return true;
    }

    Vector center;
    Vector albedo;
    double radius;
};

class LightSource {
public:

    LightSource(const Vector& position, const Vector relative_intensity=Vector(1,1,1)): position(position) {
        
        intensity[0] = MAX_LIGHT_INTENSITY * relative_intensity[0];
        intensity[1] = MAX_LIGHT_INTENSITY * relative_intensity[1];
        intensity[2] = MAX_LIGHT_INTENSITY * relative_intensity[2];
    };

    Vector position, intensity;
};

class Scene {
public:

    void addSphere(const Sphere& sphere) {
        objects.push_back(sphere);
    };

    Vector getColorIntensities(const Ray& ray, const LightSource& light_source) {

        int first_intersection_index{ 0 };
        double first_t{ std::numeric_limits<double>::max() };
        double t{ std::numeric_limits<double>::max() };
        bool intersected_once{ false };
        Vector intersection_point, intersection_normal;

        for (int i{ 0 }; i < objects.size(); i++) {

            bool intersected{ objects[i].intersect(ray, intersection_point, intersection_normal, t) };

            if (t < first_t) {
                first_t = t;
                first_intersection_index = i; // "premier" <=> le plus petit t (1er objet rencontré par le rayon)
            }

            if (intersected) intersected_once = true;
        }

        if (intersected_once) {

            objects[first_intersection_index].intersect(ray, intersection_point, intersection_normal, t);
            intersection_point = intersection_point + EPSILON * intersection_normal;

            // Lancer de rayon pour déterminer si le point d'intersection est à l'ombre de la source de lumière ou non
            double light_visibility{ 1 };
            int first_shadow_intersection_index{ 0 };
            Vector u{ light_source.position - intersection_point };
            u.normalize();
            Ray shadow_ray(intersection_point, u);
            Vector shadow_intersection_point, shadow_intersection_normal;

            // Reset des variables réutilisables pour le 2ème lancer de rayon :
            first_t = std::numeric_limits<double>::max();
            t = std::numeric_limits<double>::max();
            intersected_once = false;
            
            for (int i{ 0 }; i < objects.size(); i++) {

                bool shadow_ray_intersected{ objects[i].intersect(shadow_ray, shadow_intersection_point, shadow_intersection_normal, t) };

                if (t < first_t) {
                    first_t = t;
                    first_shadow_intersection_index = i;
                }

                if (shadow_ray_intersected) intersected_once = true;
            }

            if (intersected_once && first_t <= sqrt((light_source.position - intersection_point).norm2())) light_visibility = 0; // si intersection avant la source de lumière, pas de visibilité sur celle-ci

            double common_factor = light_visibility * dot(intersection_normal, (light_source.position - intersection_point)) / (4 * sqr(M_PI) * (light_source.position - intersection_point).norm2());
            Vector colorIntensities;
            colorIntensities[0] = light_source.intensity[0] * objects[first_intersection_index].albedo[0] * common_factor;
            colorIntensities[1] = light_source.intensity[1] * objects[first_intersection_index].albedo[1] * common_factor;
            colorIntensities[2] = light_source.intensity[2] * objects[first_intersection_index].albedo[2] * common_factor;
            return colorIntensities;
        }
        else return Vector(0, 0, 0); // couleur par défaut si pas d'intersection ciel ("sky") noir

    };

    std::vector<Sphere> objects;
};

int main() {
    int W{ 512 };
    int H{ 512 };
    double alpha{ 60 * M_PI / 180 };

    int sphere_radius{ 10 };
    int offset_to_wall{ 50 };
    int big_radius{ 100000 };

    Vector origin_camera(0, 0, 55);
    LightSource light_source(Vector(-10, 20, 40));
    Scene scene;
    scene.addSphere(Sphere(Vector(0, 0, 0), sphere_radius));
    scene.addSphere(Sphere(Vector(big_radius, 0, 0), big_radius - offset_to_wall - sphere_radius, Vector(255.0 / 255.0, 140.0 / 255.0, 0.0 / 255.0)));
    scene.addSphere(Sphere(Vector(-big_radius, 0, 0), big_radius - offset_to_wall - sphere_radius, Vector(55.0 / 255.0, 215.0 / 255.0, 0.0 / 255.0)));
    scene.addSphere(Sphere(Vector(0, big_radius, 0), big_radius - offset_to_wall - sphere_radius, Vector(238.0 / 255.0, 29.0 / 255.0, 35.0 / 255.0)));
    scene.addSphere(Sphere(Vector(0, -big_radius, 0), big_radius - sphere_radius, Vector(0.0 / 255.0, 174.0 / 255.0, 239.0 / 255.0)));
    scene.addSphere(Sphere(Vector(0, 0, -big_radius), big_radius - offset_to_wall - sphere_radius, Vector(13.0/255.0, 147.0/255.0, 68.0/255.0)));
    scene.addSphere(Sphere(Vector(0, 0, big_radius), big_radius - offset_to_wall - sphere_radius, Vector(237.0 / 255.0, 2.0 / 255.0, 140.0 / 255.0)));

    std::vector<unsigned char> image(W*H * 3, 0);

#ifdef _OPENMP
    std::cout << "OpenMP is active";
#endif

#pragma omp parallel for
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {

            Vector u(j - W / 2, H / 2 - i, - W / (2 * tan(alpha/2)));
            u.normalize();
            Ray ray(origin_camera, u);

            Vector colorIntensities = scene.getColorIntensities(ray, light_source);

            image[(i*W + j) * 3 + 0] = color_correction(colorIntensities[0]); // RED
            image[(i*W + j) * 3 + 1] = color_correction(colorIntensities[1]); // GREEN
            image[(i*W + j) * 3 + 2] = color_correction(colorIntensities[2]); // BLUE
        }
    }

    stbi_write_png("image.png", W, H, 3, &image[0], 0);
    return 0;
}
