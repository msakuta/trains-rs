//! Perlin noise implementation from Wikipedia https://en.wikipedia.org/wiki/Perlin_noise

pub(crate) fn perlin_noise_pixel(
    x: f64,
    y: f64,
    octaves: u32,
    terms: &[Terms],
    persistence: f64,
) -> f64 {
    let mut sum = 0.;
    let [mut maxv, mut f] = [0., 1.];
    for i in (0..octaves).rev() {
        let cell = 1 << i;
        let fcell = cell as f64;
        let dx = x / fcell;
        let dy = y / fcell;
        let x0 = dx.floor();
        let x1 = x0 + 1.;
        let y0 = dy.floor();
        let y1 = y0 + 1.;
        let a00 = noise_pixel(x0, y0, dx, dy, &terms[i as usize]);
        let a01 = noise_pixel(x0, y1, dx, dy, &terms[i as usize]);
        let a10 = noise_pixel(x1, y0, dx, dy, &terms[i as usize]);
        let a11 = noise_pixel(x1, y1, dx, dy, &terms[i as usize]);
        let fx = smooth_step(dx - x0);
        let fy = smooth_step(dy - y0);
        sum += ((a00 * (1. - fx) + a10 * fx) * (1. - fy) + (a01 * (1. - fx) + a11 * fx) * fy) * f;
        maxv += f;
        f *= persistence;
    }
    sum / maxv
}

pub(crate) struct Terms {
    sin_x: f64,
    sin_y: f64,
    sin_c: f64,
    cos_x: f64,
    cos_y: f64,
    cos_c: f64,
}

pub(crate) fn gen_terms(rng: &mut Xor128, bit: u32) -> Vec<Terms> {
    (0..bit)
        .map(|_| Terms {
            sin_x: 10000. * rng.next(),
            sin_y: 10000. * rng.next(),
            sin_c: std::f64::consts::PI * rng.next(),
            cos_x: 10000. * rng.next(),
            cos_y: 10000. * rng.next(),
            cos_c: std::f64::consts::PI * rng.next(),
        })
        .collect()
}

fn random_gradient(x: f64, y: f64, terms: &Terms) -> [f64; 2] {
    let random = 2920.
        * (x * terms.sin_x + y * terms.sin_y + terms.sin_c).sin()
        * (x * terms.cos_x + y * terms.cos_y + terms.cos_c).cos();
    [random.cos(), random.sin()]
}

fn smooth_step(x: f64) -> f64 {
    3. * x.powi(2) - 2. * x.powi(3)
}

fn noise_pixel(ix: f64, iy: f64, x: f64, y: f64, terms: &Terms) -> f64 {
    // Get gradient from integer coordinates
    let gradient = random_gradient(ix, iy, terms);

    // Compute the distance vector
    let dx = x - ix;
    let dy = y - iy;

    // Compute the dot-product
    dx * gradient[0] + dy * gradient[1]
}

pub(crate) struct Xor128 {
    x: u32,
    y: u32,
    z: u32,
    w: u32,
}

impl Xor128 {
    pub fn new(seed: u32) -> Self {
        let mut ret = Xor128 {
            x: 294742812,
            y: 3863451937,
            z: 2255883528,
            w: 824091511,
        };
        if 0 < seed {
            ret.x ^= seed;
            ret.y ^= ret.x;
            ret.z ^= ret.y;
            ret.w ^= ret.z;
            ret.nexti();
        }
        ret.nexti();
        ret
    }

    pub fn nexti(&mut self) -> u32 {
        // T = (I + L^a)(I + R^b)(I + L^c)
        // a = 13, b = 17, c = 5
        let t = self.x ^ (self.x << 15);
        self.x = self.y;
        self.y = self.z;
        self.z = self.w;
        self.w ^= (self.w >> 21) ^ (t ^ (t >> 4));
        self.w
    }

    pub fn next(&mut self) -> f64 {
        self.nexti() as f64 / 0xffffffffu32 as f64
    }
}
