pub struct Param {
    pub data: Vec<Vec<f32>>,
    pub grad: Vec<Vec<f32>>,
}

impl Param {
    pub fn new(data: Vec<Vec<f32>>, grad: Vec<Vec<f32>>) -> Self {
        Self { data, grad }
    }

    pub fn zero_grad(&mut self) {
        for row in &mut self.grad {
            for g in row {
                *g = 0.0;
            }
        }
    }
}
