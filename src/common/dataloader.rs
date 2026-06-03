// Sliding-window next-token dataset builder.
//
// Language-model training does not use independent labels like classification.
// It turns one long token stream into many overlapping examples:
//
//   token_ids = [10, 20, 30, 40, 50]
//   max_length = 3
//   stride = 1
//
//   item 0 input  = [10, 20, 30]
//          target = [20, 30, 40]
//
//   item 1 input  = [20, 30, 40]
//          target = [30, 40, 50]
//
// Every target row is the input shifted left by one token. This teaches the
// model: "given all previous tokens in the context, predict the next token."
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
        // Store the token stream and window settings. No examples are copied yet;
        // windows are sliced lazily by get_item()/iter().
        Self {
            token_ids,
            max_length,
            stride,
        }
    }

    pub fn len(&self) -> usize {
        if self.token_ids.len() <= self.max_length {
            // Need at least max_length input tokens plus one target token.
            0
        } else {
            // Count every valid starting position:
            //
            //   last usable input starts at:
            //     token_ids.len() - max_length - 1
            //
            // The extra "-1" reserves one token after the input window for the
            // final target element.
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
        // STEP 1: convert item index into token-stream start position.
        // stride=1 gives overlapping windows; larger stride skips ahead.
        let start = idx * self.stride;
        let end = start + self.max_length;

        // STEP 2: input window contains exactly max_length tokens.
        let input_chunk = self.token_ids[start..end].to_vec();

        // STEP 3: target window is shifted by one token.
        // This aligns each input position with the next token to predict:
        //
        //   input:  [x0, x1, x2]
        //   target: [x1, x2, x3]
        let target_chunk = self.token_ids[start + 1..end + 1].to_vec();
        (input_chunk, target_chunk)
    }

    /// Create an idiomatic Rust Iterator to step through the dataset.
    pub fn iter(&self) -> DataLoaderIterator<'_> {
        // The iterator borrows this loader and stores only the next item index.
        // It does not clone the full dataset.
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
            // Yield the current window pair, then advance to the next starting
            // position for the following call.
            let item = self.loader.get_item(self.current_idx);
            self.current_idx += 1;
            Some(item)
        } else {
            None
        }
    }
}
