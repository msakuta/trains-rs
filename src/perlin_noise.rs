//! Perlin noise implementation from Wikipedia https://en.wikipedia.org/wiki/Perlin_noise
//! with Xorshift64Star RNG to allow fast and uniform randomness https://en.wikipedia.org/wiki/Xorshift

pub type Seed = u64;

pub(crate) fn perlin_noise_pixel(
    x: f64,
    y: f64,
    octaves: u32,
    seeds: &[Seed],
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
        let a00 = noise_pixel(x0, y0, dx, dy, seeds[i as usize]);
        let a01 = noise_pixel(x0, y1, dx, dy, seeds[i as usize]);
        let a10 = noise_pixel(x1, y0, dx, dy, seeds[i as usize]);
        let a11 = noise_pixel(x1, y1, dx, dy, seeds[i as usize]);
        let fx = smooth_step(dx - x0);
        let fy = smooth_step(dy - y0);
        sum += ((a00 * (1. - fx) + a10 * fx) * (1. - fy) + (a01 * (1. - fx) + a11 * fx) * fy) * f;
        maxv += f;
        f *= persistence;
    }
    sum / maxv
}

pub(crate) fn gen_seeds(rng: &mut Xorshift64Star, bit: u32) -> Vec<Seed> {
    (0..bit).map(|_| rng.nexti()).collect()
}

fn i64_to_u64(i: i64) -> u64 {
    if i < 0 {
        i.wrapping_sub(i64::MIN) as u64 + i64::MAX as u64
    } else {
        i as u64
    }
}

fn random_gradient(x: f64, y: f64, seed: Seed) -> [f64; 2] {
    // Mind the overflow!
    let mut rng = Xorshift64Star::new(
        i64_to_u64(x as i64)
            .wrapping_mul(3125)
            .wrapping_add(i64_to_u64(y as i64).wrapping_mul(5021904))
            .wrapping_add(seed.wrapping_mul(9650143)),
    );
    // rng.nexti();
    let angle = rng.next() * std::f64::consts::PI * 2.;
    [angle.cos(), angle.sin()]
}

fn smooth_step(x: f64) -> f64 {
    3. * x.powi(2) - 2. * x.powi(3)
}

fn noise_pixel(ix: f64, iy: f64, x: f64, y: f64, seed: Seed) -> f64 {
    // Get gradient from integer coordinates
    let gradient = random_gradient(ix, iy, seed);

    // Compute the distance vector
    let dx = x - ix;
    let dy = y - iy;

    // Compute the dot-product
    dx * gradient[0] + dy * gradient[1]
}

pub(crate) struct Xorshift64Star {
    a: u64,
}

impl Xorshift64Star {
    pub fn new(seed: u64) -> Self {
        Self {
            a: if seed == 0 { 349001341 } else { seed },
        }
    }

    pub fn nexti(&mut self) -> u64 {
        let mut x = self.a;
        x ^= x << 12;
        x ^= x >> 25;
        x ^= x << 27;
        self.a = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    pub fn next(&mut self) -> f64 {
        self.nexti() as f64 / u64::MAX as f64
    }
}

pub fn white_noise(x: f64, y: f64, seed: Seed) -> f64 {
    let mut rng = Xorshift64Star::new(
        i64_to_u64(x as i64)
            .wrapping_mul(3125)
            .wrapping_add(i64_to_u64(y as i64).wrapping_mul(5021904))
            .wrapping_add(seed.wrapping_mul(9650143)),
    );
    rng.next()
}

pub fn white_fractal_noise(x: f64, y: f64, seeds: &[Seed], persistence: f64) -> f64 {
    let mut sum = 0.;
    let [mut maxv, mut f] = [0., 1.];
    for (i, seed) in seeds.iter().enumerate().rev() {
        let cell = 1 << i;
        let fcell = cell as f64;
        let dx = x / fcell;
        let dy = y / fcell;
        let x0 = dx.floor();
        let x1 = x0 + 1.;
        let y0 = dy.floor();
        let y1 = y0 + 1.;
        let a00 = white_noise(x0, y0, *seed);
        let a01 = white_noise(x0, y1, *seed);
        let a10 = white_noise(x1, y0, *seed);
        let a11 = white_noise(x1, y1, *seed);
        let fx = smooth_step(dx - x0);
        let fy = smooth_step(dy - y0);
        sum += ((a00 * (1. - fx) + a10 * fx) * (1. - fy) + (a01 * (1. - fx) + a11 * fx) * fy) * f;
        maxv += f;
        f *= persistence;
    }
    sum / maxv
}
