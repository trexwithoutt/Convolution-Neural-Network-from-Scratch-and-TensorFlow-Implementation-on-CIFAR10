import numpy as np

DEBUG = True

class relu:
    
    def function(self, X):
        '''Return elementwise relu values'''
        return(np.maximum(np.zeros(X.shape), X))
    
    def derivative(self, X):
        # An example of broadcasting
        return((X >= 0).astype(int))
        
class no_act:
    """Implement a no activation function and derivative"""
    
    def function(self, X):
        return(X)
    
    def derivative(self, X):
        return(np.ones(X.shape))

# Set of allowed/implemented activation functions
ACTIVATIONS = {'relu': relu,
               'no_act': no_act}    
    
class CNNLayer:
    """
    Implement a class that processes a single CNN layer.
    
    Let i be the index on the neurons in the ith layer, and j be the index on the 
    nuerons in the next outler layer.  (Following Russell-Norvig notation.) Implement 
    the following:
    
    0. __init__: Initalize filters.
    
    1. forward step: Input a_i values.  Output a_j values.  Make copies of a_i values 
       and in_j values since needed in backward_step and filter_gradient.
    
    2. backward_step: Input (del L)/(del a_j) values.  Output (del L)/(del a_i).
    
    3. filter_gradient: Input (del L)/(del a_j) values. Output (del L)/(del w_{ij}) values.
    
    4. update: Given learning rate, update filter weights. 
    
    """
    
    def __init__(self, n, filter_shape, activation='no_act', stride = 1):
        """
        Initialize filters.
        
        filter_shape is (width of filter, height of filter, depth of filter). Depth 
        of filter should match depth of the forward_step input X.
        """
        
        self.num_filters = n
        self.stride = stride
        self.filter_shape = filter_shape
        try:
            self.filter_height = filter_shape[0]
            self.filter_width = filter_shape[1]
            self.filter_depth = filter_shape[2]
        except:
            raise Exception(f'Unexpected filter shape {filter_shape}')
        try:
            # Create an object of the activation class
            self.activation = ACTIVATIONS[activation]() 
        except:
            raise Exception(f'Unknown activation: {activation}')
        self.filters = self.filters_init()
        self.biases = self.biases_init()
        self.num_examples = None 
        # Set num_of_examples during forward step, and use to verify
        # consistency during backward step.  Similarly the data height, 
        # width, and depth.
        self.data_height = None
        self.data_width = None
        self.data_depth = None
        self.data_with_pads = None
        self.in_j = None  # the in_j values for next layer.
        
    def filters_init(self):
        return np.random.random((self.num_filters, self.filter_height,
                                 self.filter_width, self.filter_depth))
    
    def biases_init(self):
        return np.random.random(self.num_filters)
    
    def set_filters(self, filters, biases):
        """Set filters to given weights.
        
           Useful in debugging."""
        if filters.shape != (self.num_filters, self.filter_height,
                                 self.filter_width, self.filter_depth):
            raise Exception(f'Mismatched filter shapes: stored '
                            f'{self.num_filters} {self.filter_shape} vs '
                            f'{filters.shape}.')
        if biases.shape != (self.num_filters,):
            raise Exception((f'Mismatched biases: stored '
                             f'{self.num_filters} vs '
                             f'{biases.shape}.'))
        self.filters = filters.copy()
        self.biases = biases.copy()
        
    def forward_step(self, X, pad_height=0, pad_width=0):
        """
        Implement a forward step.
        
        X.shape is (number of examples, height of input, width of input, depth of input).
        """
        
        try:
            # Store shape values to verify consistency during backward step
            self.num_examples = X.shape[0]
            self.data_height = X.shape[1]
            self.data_width = X.shape[2]
            self.data_depth = X.shape[3]
        except:
            raise Exception(f'Unexpected data shape {X.shape}')
        if self.data_depth != self.filter_depth:
            raise Exception(f'Depth mismatch: filter depth {self.filter_depth}'
                            f' data depth {self.data_depth}')
        self.pad_height = pad_height
        self.pad_width = pad_width
        self.input_height = self.data_height + 2 * self.pad_height
        self.input_width = self.data_width + 2 * self.pad_width
        
        # Add pad to X.  Only add pads to the 1, 2 (ht, width) axes of X, 
        # not to the 0, 4 (num examples, depth) axes.
        # 'constant' implies 0 is added as pad.
        X = np.pad(X, ((0,0),(pad_height, pad_height), 
                      (pad_width, pad_width), (0,0)), 'constant')
        
        # Save this for the update step
        # self.data_with_pads = X.copy() #REMOVE THIS
        # Save a copy for computing filter_gradient
        self.a_i = X.copy()  #
        
        # Get height, width after padding
        height = X.shape[1]
        width = X.shape[2]

        # Don't include pad in formula because height includes it.
        output_height = ((height - self.filter_height)/self.stride + 1)
        output_width = ((width - self.filter_width)/self.stride + 1)    
        if (
            output_height != int(output_height) or 
            output_width != int(output_width)
        ):
            raise Exception(f"Filter doesn't fit: {output_height} x {output_width}")
        else:
            output_height = int(output_height)
            output_width = int(output_width)
            
        #####################################################################
        # There are two ways to convolve the filters with X.
        # 1. Using the im2col method described in Stanford 231 notes.
        # 2. Using NumPy's tensordot method.
        #
        # (1) requires more code.  (2) requires understanding how tensordot
        # works.  Most likely tensordot is more efficient.  To illustrate both,
        # in the code below data_tensor is constructed using (1) and 
        # new_data_tensor is constructed using (2).  You may use either.
            
        # Stanford's im2col method    
        # Construct filter tensor and add biases
        filter_tensor = self.filters.reshape(self.num_filters, -1)
        
        filter_tensor = np.hstack((self.biases.reshape((-1,1)), filter_tensor))
        # Construct the data tensor
        # The im2col_length does not include the bias terms
        # Biases are later added to both data and filter tensors
        im2col_length = self.filter_height * self.filter_width * self.filter_depth
        num_outputs = output_height * output_width
        data_tensor = np.empty((self.num_examples, num_outputs, im2col_length))
        for h in range(output_height):
            for w in range(output_width):
                hs = h * self.stride
                ws = w * self.stride
                data_tensor[:,h*output_width + w, :] = X[:,hs:hs+self.filter_height,
                                    ws:ws+self.filter_width,:].reshape(
                                        (self.num_examples,-1))  
        # add bias-coeffs to data tensor
        data_tensor = np.concatenate((np.ones((self.num_examples, num_outputs, 1)),
                                 data_tensor), axis=2)
        output_tensor = np.tensordot(data_tensor, filter_tensor, axes=([2],[1]))
        output_tensor = output_tensor.reshape(
            (self.num_examples,output_height,output_width,self.num_filters))
        
        
        
        # NumPy's tensordot based method
        new_output_tensor = np.empty((self.num_examples, output_height, 
                                      output_width, self.num_filters))
        for h in range(output_height):
            for w in range(output_width):
                hs = h * self.stride
                ws = w * self.stride
                new_output_tensor[:,h,w,:] = np.tensordot(
                                                X[:, # example
                                                  hs:hs+self.filter_height, # height
                                                  ws:ws+self.filter_width,  # width
                                                  : # depth
                                                ], 
                                                self.filters[:, # filter 
                                                             :, # height
                                                             :, # width
                                                             :  # depth
                                                ], 
                                                axes = ((1,2,3),(1,2,3))
                                              )
                # Add bias term
                new_output_tensor[:,h,w,:] = (new_output_tensor[:,h,w,:] + 
                                              self.biases)
        # Check both methods give the same answer
        assert np.array_equal(output_tensor, new_output_tensor)
                
        
        self.in_j = output_tensor.copy() # Used in backward_step.
        output_tensor = self.activation.function(output_tensor) # a_j values
        return(output_tensor)
      
    def backward_step(self, D):
        """
        Implement the backward step and return (del L)/(del a_i). 
        
        Given D=(del L)/(del a_j) values return (del L)/(del a_i) values.  
        D (delta) is of shape (number of examples, height of output (i.e., the 
        a_j values), width of output, depth of output)"""
                
        try:
            num_examples = D.shape[0]
            delta_height = D.shape[1]
            delta_width = D.shape[2]
            delta_depth = D.shape[3]
        except:
            raise Exception(f'Unexpected delta shape {D.shape}')
        if num_examples != self.num_examples:
            raise Exception(f'Number of examples changed from forward step: '
                             f'{self.num_examples} vs {num_examples}')
        if delta_depth != self.num_filters:
            raise Exception(f'Depth mismatch: number of filters {self.num_filters}' 
                            f' delta depth {delta_depth}')
        # Make a copy so that we can change it
        prev_delta = D.copy()
        if prev_delta.ndim != 4:
            raise Exception(f'Unexpected number of dimensions {D.ndim}')
        new_delta = None
        
        ####################################################################
        # WRITE YOUR CODE HERE
        D = D * self.activation.derivative(self.in_j)
        rotated_filter = np.rot90(self.filters, 2, (1,2))
        new_del_h = self.data_height + self.pad_height * 2
        new_del_w = self.data_width + self.pad_height * 2
        new_del_d = self.data_depth
        new_del = np.empty((self.num_examples, new_del_h, 
                              new_del_w, new_del_d))
        
        pad_size = (new_del_h + self.filter_height - delta_height - 1) / 2
        
        if pad_size != int(pad_size):
            raise Exception('Padding shape:'
                            'padding cannnot be added')
            
        pad_size = int(pad_size)
        
        D = np.pad(D,
                   ((0,0),(pad_size, pad_size), (pad_size, pad_size), (0,0)),
                   'constant')
        
        for h in range(new_del_h):
            for w in range(new_del_w):
                new_del[:,h,w,:] = np.tensordot(
                                                  D[:,
                                                  h:h+self.filter_height,
                                                  w:w+self.filter_width,
                                                  :
                                                ], 
                                                rotated_filter[:, # filter 
                                                               :, # height
                                                               :, # width
                                                               :  # depth
                                                ], 
                                                axes = ((1,2,3),(1,2,0))
                )
        return(new_del)
    
    def filter_gradient(self, D):
        """
        Return the filter_gradient.
        
        D = (del L)/(del a_j) has shape (num_examples, height, width, depth=num_filters)
        The filter_gradient (del L)/(del w_{ij}) has shape (num_filters, filter_height, 
        filter_width, filter_depth=input_depth)
        
        """
         
        if DEBUG and D.ndim != 4:
            raise Exception(f'D has {D.ndim} dimensions instead of 4.')
        # D depth should match number of filters
        D_depth = D.shape[3]
        if DEBUG:
            if D_depth != self.num_filters:
                raise Exception(f'D depth {D_depth} != num_filters'
                                f' {self.num_filters}')
            if D.shape[0] != self.num_examples:
                raise Exception(f'D num_examples {D.shape[0]} !='
                                f'num_examples {self.num_examples}')
        f_gradient = None
        
        ####################################################################
        # WRITE YOUR CODE HERE
        D = D * self.activation.derivative(self.in_j)
        f_gradient = np.empty((self.num_filters, self.filter_height,
                               self.filter_width, self.filter_depth))
        
        pad_size = (self.filter_height + self.a_i.shape[1] - D.shape[1] - 1) / 2
        
        if pad_size != int(pad_size):
            raise Exception('Padding shape:'
                            'padding cannnot be added')
            
        pad_size = int(pad_size)
        
        D = np.pad(D,
                   ((0,0),(pad_size, pad_size),(pad_size, pad_size), (0,0)),
                   'constant')
        for h in range(self.filter_height):
            for w in range(self.filter_width):
                f_gradient[:,h,w,:] = np.tensordot(
                                                  D[:,
                                                    h:h+self.a_i.shape[1],
                                                    w:w+self.a_i.shape[2],
                                                    :
                                                   ], 
                                                self.a_i[:, # example 
                                                         :, # height
                                                         :, # width
                                                         :  # depth
                                                        ], 
                                                axes = ((0,1,2),(0,1,2))
                )
        f_gradient = np.rot90(f_gradient, 2, (1,2))
        return(f_gradient)