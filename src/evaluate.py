from data.schema import ExtractionResult
from src.generate import (
    extract,
    unconstrained_extract,
    build_generator,
    load_hf_model,
    build_chat_prompt,
)
from src.chunking.spans import Chunk


def validate(Chunk) -> ExtractionResult:
    from pydantic import ValidationError

    model, tokenizer = load_hf_model()
    generator = build_generator(model, tokenizer)
    raw = extract(generator, tokenizer, Chunk)

    try:
        result = ExtractionResult.model_validate(raw)
        valid = True
    except ValidationError as error:
        print(error)
        valid = False
    finally:
        return valid


"""
Validity                                                                                                                               
                                                                                                                                        
 The README defines it as the fraction of outputs that parse as JSON and validate against ExtractionResult, over all eval chunks.       
                                                                                                                                        
 1. Take the raw text from extract or unconstrained_extract.                                                                            
 2. Try to turn it into an ExtractionResult. The model_validate_json method on a Pydantic model does both the parse and the schema      
    check in one call. Per the Pydantic JSON docs, it raises ValidationError on bad input.                                              
 3. Success counts as valid. An exception counts as invalid.                                                                            
 4. Validity rate = valid outputs ÷ all eval chunks.                                                                                    
                                                                                                                                        
 Your schema is strict: node types and relations are enums, and names must match the identifier regex. So the validator also catches    
 invented relation types and multi-word names.                                                                                          
                                                                                                                                        
 A truncated JSON (generation stopped at max_new_tokens) fails to parse. The README says to count it as invalid, not skip it and not    
 let it crash the run.        

 Precision and recall                                                                                                                   
                                                                                                                                        
 1. For each valid output, build a set of triples: {(subject, relation, target), ...}. Do the same for the hand-corrected reference.    
 2. Compare them:                                                                                                                       
   - TP (true positives) = triples in both sets, pred & gold                                                                            
   - FP (false positives) = predicted but not in the reference, pred - gold                                                             
   - FN (false negatives) = in the reference but missed, gold - pred                                                                    
 3. Then:                                                                                                                               
   - Precision = TP ÷ (TP + FP): of what the model said, how much was right.                                                            
   - Recall = TP ÷ (TP + FN): of what was there, how much the model found.                                                              

"""
