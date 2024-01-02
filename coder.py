import numpy as np

class ClassCoder:
    def __init__(self, class_encoder, set_classes, taxonomy=None):
        self.class_encoder = class_encoder
        self.class_decoder = {i: c for c, i in self.class_encoder.items()}
        self.class_decoder[len(set_classes)] = 'unknown'
        self.taxonomy = taxonomy

        if taxonomy is not None:
            self.species2sci = {row.SPECIES_CODE: row.SCI_NAME for _, row in taxonomy.iterrows()}
            self.species2sci['noise'] = 'noise'
            self.species2order = {row.SPECIES_CODE: row.ORDER1 for _, row in taxonomy.iterrows()}
            self.species2order['noise'] = 'noise'
            self.species2family = {row.SPECIES_CODE: row.FAMILY for _, row in taxonomy.iterrows()}
            self.species2family['noise'] = 'noise'

            # assign to scientific name a number
            self.sci2code = {k: i for i, k in enumerate(sorted(set(self.species2sci.values())))}
            self.order2code = {k: i for i, k in enumerate(sorted(taxonomy['ORDER1'].dropna().unique()))}
            self.order2code['noise'] = len(self.order2code)
            self.family2code = {k: i for i, k in enumerate(sorted(taxonomy['FAMILY'].dropna().unique()))}
            self.family2code['noise'] = len(self.family2code)

            #code to scientific name
            self.code2sci = {i: k for k, i in self.sci2code.items()}
            self.code2order = {i: k for k, i in self.order2code.items()}
            self.code2family = {i: k for k, i in self.family2code.items()}

    def spec_to_sci(self, spec_code):
        if isinstance(spec_code, (list, tuple, np.ndarray)): 
            return [self.species2sci.get(code, 'unknown') for code in spec_code]
        return self.species2sci.get(spec_code, 'unknown')

    def spec_to_order(self, spec_code):
        if isinstance(spec_code, (list, tuple, np.ndarray)): 
            return [self.species2order.get(code, 'unknown') for code in spec_code]
        return self.species2order.get(spec_code, 'unknown')
    
    def spec_to_family(self, spec_code):
        if isinstance(spec_code, (list, tuple, np.ndarray)): 
            return [self.species2family.get(code, 'unknown') for code in spec_code]
        return self.species2family.get(spec_code, 'unknown')

    def encode_sci(self, sci_name):
        if isinstance(sci_name, (list, tuple, np.ndarray)): 
            return [self.sci2code.get(name, len(self.sci2code)) for name in sci_name]
        return self.sci2code.get(sci_name, len(self.sci2code))
    
    def decode_sci(self, idx):
        if isinstance(idx, (list, tuple, np.ndarray)):  
            return [self.code2sci.get(i, 'unknown') for i in idx]
        return self.code2sci.get(idx, 'unknown')
    
    def encode_order(self, order_name):
        if isinstance(order_name, (list, tuple, np.ndarray)): 
            return [self.order2code.get(name, len(self.order2code)) for name in order_name]
        return self.order2code.get(order_name, len(self.order2code))
    
    def decode_order(self, idx):
        if isinstance(idx, (list, tuple, np.ndarray)):  
            return [self.code2order.get(i, 'unknown') for i in idx]
        return self.code2order.get(idx, 'unknown')
    
    def encode_family(self, family_name):
        if isinstance(family_name, (list, tuple, np.ndarray)): 
            return [self.family2code.get(name, len(self.family2code)) for name in family_name]
        return self.family2code.get(family_name, len(self.family2code))
    
    def decode_family(self, idx):
        if isinstance(idx, (list, tuple, np.ndarray)):  
            return [self.code2family.get(i, 'unknown') for i in idx]
        return self.code2family.get(idx, 'unknown')

    def encode(self, class_name):
        if isinstance(class_name, (list, tuple, np.ndarray)): 
            return [self.class_encoder.get(name, len(self.class_decoder) - 1) for name in class_name]
        return self.class_encoder.get(class_name, len(self.class_decoder) - 1) 

    def decode(self, idx):
        if isinstance(idx, (list, tuple, np.ndarray)):  
            return [self.class_decoder.get(i, 'unknown') for i in idx]
        return self.class_decoder.get(idx, 'unknown')  

    def __len__(self):
        return len(self.class_decoder)
