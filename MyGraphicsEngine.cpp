#define _CRT_SECURE_NO_WARNINGS 1
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define M_PI 3.14159265358

#include <iostream>

static inline double sqr(double x) { return x * x; }

unsigned char correc_pixel_value(double num) {

    num = pow(num, 1/2.2); // correction gamma
    if (num > 255) return 255; // clamping
    else if (num < 0) return 0; // clamping
    else return static_cast<unsigned char>(num);

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

    bool intersect(const Ray& ray, Vector& intersection_point, Vector& normal_vector) {
        double a{ 1 };
        double b = 2 * dot(ray.direction, ray.origin - center);
        double c = (ray.origin - center).norm2() - sqr(radius);

        double delta = sqr(b) - 4 * a * c;

        if (delta < 0) return false; // pas d'intersection

        double sqrt_delta = sqrt(delta);
        double t1 = (-b - sqrt_delta) / (2 * a);
        double t2 = (-b + sqrt_delta) / (2 * a);

        if (t2 < 0) return false; // intersection pas du bon côté de la caméra

        double t;

        if (t1 >= 0) {
            t = t1;
        } else {
            t = t2; // la caméra est dans la sphère
        }

        intersection_point = ray.origin + t * ray.direction; // point d'intersection
        normal_vector = intersection_point - center; // vecteur normal à la sphère en intersection_point
        normal_vector.normalize();

        return true;
    }

    Vector center;
    Vector albedo;
    double radius;

};

class LightSource {
public:

    LightSource(const Vector& position, const Vector intensity=Vector(1,1,1)): position(position) {
        
        this->intensity[0] = max_intensity * intensity[0];
        this->intensity[1] = max_intensity * intensity[1];
        this->intensity[2] = max_intensity * intensity[2];
    };

    const double max_intensity{ 2e8 };
    Vector position, intensity;

};


int main() {
    int W = 512;
    int H = 512;
    double alpha = 60*M_PI/180;

    Vector origin_camera(0, 0, 55);
    Sphere sphere(Vector(0,0,0), 10);
    LightSource light_source(Vector(-10, 20, 40));

    std::vector<unsigned char> image(W*H * 3, 0);

#ifdef _OPENMP
    std::cout << "OpenMP is active";
#endif

#pragma omp parallel for
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {

            Vector u(j - W / 2, H / 2 - i, - W / (2 * tan(alpha/2)));
            u.normalize();
            Ray r(origin_camera, u);

            Vector P;
            Vector N;

            unsigned char red{ 0 };
            unsigned char green{ 0 };
            unsigned char blue{ 0 };

            if (sphere.intersect(r, P, N)) {

                double scal_prod = dot(N, (light_source.position - P)) / (light_source.position - P).norm2();

                red = correc_pixel_value(light_source.intensity[0] * sphere.albedo[0] * scal_prod / (4 * sqr(M_PI)));
                green = correc_pixel_value(light_source.intensity[1] * sphere.albedo[1] * scal_prod / (4 * sqr(M_PI)));
                blue = correc_pixel_value(light_source.intensity[2] * sphere.albedo[2] * scal_prod / (4 * sqr(M_PI)));
            }

            image[(i*W + j) * 3 + 0] = red; // RED
            image[(i*W + j) * 3 + 1] = green; // GREEN
            image[(i*W + j) * 3 + 2] = blue; // BLUE
        }
    }
    stbi_write_png("image.png", W, H, 3, &image[0], 0);

    return 0;
}
