pub struct DataLoader {
    /// The full tokenized dataset
    pub token_ids: Vec<usize>,
    /// Size of the input context window
    pub max_length: usize,
    /// Distance the window slides at each step
    pub stride: usize,
}

impl DataLoader {
    pub fn new(token_ids: Vec<usize>, max_length: usize, stride: usize) -> Self {
        Self {
            token_ids,
            max_length,
            stride,
        }
    }

    pub fn len(&self) -> usize {
        if self.token_ids.len() <= self.max_length {
            0
        } else {
            // formula of total batch no of corpus, accounting for next-token target shift (+1)
            (self.token_ids.len() - self.max_length - 1) / self.stride + 1
        }
    }


    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Retrieve a specific input-target window pair by index.
    ///
    /// - Inputs: token_ids[start .. start + max_length]
    /// - Targets: token_ids[start + 1 .. start + max_length + 1]
    pub fn get_item(&self, idx: usize) -> (Vec<usize>, Vec<usize>) {
        let start = idx * self.stride;
        let end = start + self.max_length;
        let input_chunk = self.token_ids[start..end].to_vec();
        let target_chunk = self.token_ids[start + 1..end + 1].to_vec();
        (input_chunk, target_chunk)
    }
    /// Create an idiomatic Rust Iterator to step through the dataset.
    pub fn iter(&self) -> DataLoaderIterator<'_> {
        DataLoaderIterator {
            loader: self,
            current_idx: 0,
        }
    }
}

/// An Iterator that slides over the DataLoader dataset chunk-by-chunk.
pub struct DataLoaderIterator<'a> {
    loader: &'a DataLoader,
    current_idx: usize,
}

impl<'a> Iterator for DataLoaderIterator<'a> {
    /// Each step yields an (inputs, targets) tuple of Vecs
    type Item = (Vec<usize>, Vec<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx < self.loader.len() {
            let item = self.loader.get_item(self.current_idx);
            self.current_idx += 1;
            Some(item)
        } else {
            None
        }
    }
}
