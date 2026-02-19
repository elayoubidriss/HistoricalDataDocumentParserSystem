import base64
from pydantic import BaseModel
from enum import Enum
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import os
from transformers import AutoTokenizer
from transformers import AutoTokenizer
from collections import defaultdict
from typing import List, Tuple, Optional, Any


class CoordType(Enum):
    UNSTRUCTURED = "unstructured"
    DOCLING = "docling"

class Point(BaseModel):
    x: float
    y: float

class CoordinatesFromTopLeft(BaseModel):
    top_left: Point
    top_right: Point
    bottom_left: Point
    bottom_right: Point


def process_coordinates(elements, coords_type, file_path,file_name):
    result = []
    if elements:
        for element in elements :
            element[2] = convert_to_coords(element[2],coords_type,os.path.join(file_path, file_name),element[3])
            result.append(element)
    return result

def extend_coordinates(elements, file_path , file_name, crop_extension):
    result = []
    if elements:
        for element in elements :
            element[2] = extend_coords(pdf_path=os.path.join(file_path, file_name),coordinates=element[2],crop_extension=crop_extension,page_number=element[3])
            result.append(element)
    return result


def get_page_height(document_path: str, page_number: int) -> float:
    """Extract page height from PDF document."""
    doc = fitz.open(document_path)
    try:
        page = doc[page_number]
        return page.rect.height
    finally:
        doc.close()

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
def convert_to_coords(coords, coord_type: CoordType = CoordType.UNSTRUCTURED, 
                     document_path: Optional[str] = None, page_number: Optional[int] = None):
    if coord_type == CoordType.UNSTRUCTURED:
        # Existing logic for unstructured coordinates
        top_left, bottom_left, bottom_right, top_right = coords
        
        return CoordinatesFromTopLeft(
            bottom_left=Point(x=bottom_left[0], y=bottom_left[1]),
            top_left=Point(x=top_left[0], y=top_left[1]),
            top_right=Point(x=top_right[0], y=top_right[1]),
            bottom_right=Point(x=bottom_right[0], y=bottom_right[1])
        )
    
    elif coord_type == CoordType.DOCLING:
        # Handle BoundingBox with BOTTOMLEFT origin
        if document_path is None or page_number is None:
            raise ValueError("document_path and page_number are required for DOCLING coordinate conversion")
        
        page_height = get_page_height(document_path, page_number-1)
        
        # Convert from bottom-left origin to top-left origin
        # In bottom-left: t is top, b is bottom
        # In top-left: we need to flip the y-coordinates
        top_y = page_height - coords.t  # top in top-left origin
        bottom_y = page_height - coords.b  # bottom in top-left origin
        
        return CoordinatesFromTopLeft(
            top_left=Point(x=coords.l, y=top_y),
            top_right=Point(x=coords.r, y=top_y),
            bottom_right=Point(x=coords.r, y=bottom_y),
            bottom_left=Point(x=coords.l, y=bottom_y)
        )


def get_tokenizer(tokenization_model) : 
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenization_model)
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")

def get_tokens_number(text,tokenization_model):
    tokenizer = get_tokenizer(tokenization_model)
    tokens = tokenizer.encode(text)
    return len(tokens)

def chunk_text(text, max_tokens, overlap_tokens, tokenization_model):
    
    tokenizer = get_tokenizer(tokenization_model)
   
    if not text.strip():
        return []
    
    # Encode the entire text to get token IDs
    token_ids = tokenizer.encode(text)
    token_numbers = len(token_ids)
    
    # If text is shorter than max_tokens, return as single chunk
    if token_numbers <= max_tokens:
        return [text]
    
    chunks = []
    start_idx = 0
    
    while start_idx < token_numbers:
        # Calculate end index for this chunk
        end_idx = min(start_idx + max_tokens, len(token_ids))
        
        # Get chunk of token IDs
        chunk_token_ids = token_ids[start_idx:end_idx]
        
        # Decode back to text
        chunk_text = tokenizer.decode(chunk_token_ids)
        chunks.append(chunk_text)
        
        # Move start index forward, accounting for overlap
        start_idx = end_idx - overlap_tokens
        
        # Break if we've reached the end
        if end_idx >= len(token_ids):
            break
    
    return chunks

def extend_coords(pdf_path: str, coordinates: CoordinatesFromTopLeft, crop_extension: float, page_number: int) -> CoordinatesFromTopLeft:
    
    doc   = fitz.open(pdf_path)
    page  = doc[page_number - 1]

    # 1. build a fitz.Rect in *points* (top-left origin)
    tl, tr, br, bl = (coordinates.top_left, coordinates.top_right,
                      coordinates.bottom_right, coordinates.bottom_left)

    bbox = fitz.Rect(tl.x, tl.y, br.x, br.y)

    # 2. optional padding in point units
    w, h = bbox.width, bbox.height
    ext_x, ext_y = w * crop_extension, h * crop_extension
    bbox.x0 = max(0, bbox.x0 - ext_x)
    bbox.y0 = max(0, bbox.y0 - ext_y)
    bbox.x1 = min(page.rect.x1, bbox.x1 + ext_x)
    bbox.y1 = min(page.rect.y1, bbox.y1 + ext_y)

    # 3. Create extended coordinates from the bbox
    extended_coords = CoordinatesFromTopLeft(
        top_left=Point(x=bbox.x0, y=bbox.y0),
        top_right=Point(x=bbox.x1, y=bbox.y0),
        bottom_right=Point(x=bbox.x1, y=bbox.y1),
        bottom_left=Point(x=bbox.x0, y=bbox.y1)
    )

    doc.close()  # Don't forget to close the document
    
    return extended_coords


def crop_and_encode_image(pdf_path: str, coordinates: CoordinatesFromTopLeft, page_number: int) -> str:
    doc   = fitz.open(pdf_path)
    page  = doc[page_number - 1]

    # 1. build a fitz.Rect in *points* (top-left origin)
    tl, tr, br, bl = (coordinates.top_left, coordinates.top_right,
                      coordinates.bottom_right, coordinates.bottom_left)

    bbox = fitz.Rect(tl.x, tl.y, br.x, br.y)

    # 3. get_pixmap renders *only* that rectangle at 300 dpi
    pix = page.get_pixmap(clip=bbox, dpi=300)

    img_b64 = base64.b64encode(pix.tobytes("png")).decode()
    
    return img_b64

def get_bounding_rectangle_multiple(rectangles):
    """
    Calculate the bounding rectangle that encompasses multiple rectangles.
    
    Args:
        rectangles: List of CoordinatesFromTopLeft objects
        
    Returns:
        CoordinatesFromTopLeft object representing the bounding rectangle
    """
    if not rectangles:
        return None
    
    # Extract all coordinates from all rectangles
    all_x_coords = []
    all_y_coords = []
    
    for rect in rectangles:
        # Extract coordinates from CoordinatesFromTopLeft object
        coords = [rect.top_left, rect.top_right, rect.bottom_right, rect.bottom_left]
        all_x_coords.extend([coord.x for coord in coords])
        all_y_coords.extend([coord.y for coord in coords])
    
    # Find min and max x and y values
    min_x = min(all_x_coords)
    max_x = max(all_x_coords)
    min_y = min(all_y_coords)
    max_y = max(all_y_coords)
    
    # Return as CoordinatesFromTopLeft object
    return CoordinatesFromTopLeft(
        top_left=Point(x=min_x, y=min_y),
        top_right=Point(x=max_x, y=min_y),
        bottom_right=Point(x=max_x, y=max_y),
        bottom_left=Point(x=min_x, y=max_y)
    )

def split_texts_by_tokens(elements, max_tokens, overlap_tokens,tokenization_model):
    """
    Chunk text elements based on token limits while maintaining page boundaries.
    
    Args:
        elements: List in format:
                ['Text', content, CoordinatesFromTopLeft, page_number]
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Number of overlapping tokens for text splitting
    
    Returns:
        List of chunked elements with combined text and bounding coordinates
    """
    if not elements:
        return []
    
    chunks = []
    current_chunk_elements = []
    current_chunk_text = ""
    current_page = None
    
    def create_chunk_from_elements(chunk_elements):
        """Helper function to create a chunk from accumulated elements"""
        if not chunk_elements:
            return None
            
        # Combine all text
        combined_text = " ".join([elem[1] for elem in chunk_elements])
        
        # Calculate bounding coordinates
        rectangles = [elem[2] for elem in chunk_elements]
        bounding_coords = get_bounding_rectangle_multiple(rectangles)
        
        # Use the page number from the first element (they should all be the same)
        page_number = chunk_elements[0][3] if len(chunk_elements[0]) > 3 else None
        
        return ['Text', combined_text, bounding_coords, page_number]
    
    def add_chunk_to_results(chunk_elements):
        """Add current chunk to results and reset"""
        chunk = create_chunk_from_elements(chunk_elements)
        if chunk:
            chunks.append(chunk)
    
    for element in elements:
        element_type, text, coords, page_num = element
        
        # Check if this element's text exceeds token limit by itself
        element_tokens = get_tokens_number(text, tokenization_model)
        
        if element_tokens > max_tokens:
            # First, add any existing chunk
            if current_chunk_elements:
                add_chunk_to_results(current_chunk_elements)
                current_chunk_elements = []
                current_chunk_text = ""
            
            # Split the large text into smaller chunks
            text_chunks = chunk_text(text, max_tokens, overlap_tokens,tokenization_model)
            
            # Create separate elements for each text chunk
            for text_chunk in text_chunks:
                chunk_element = [element_type, text_chunk, coords, page_num]
                chunks.append(chunk_element)
            
            current_page = page_num
            continue
        
        # Check if we should start a new chunk
        should_start_new_chunk = False
        
        if current_chunk_elements:
            # Different page number
            if page_num != current_page:
                should_start_new_chunk = True
            else:
                # Same page, check if adding this element would exceed token limit
                potential_text = current_chunk_text + " " + text
                potential_tokens = get_tokens_number(potential_text, tokenization_model)
                if potential_tokens > max_tokens:
                    should_start_new_chunk = True
        
        if should_start_new_chunk:
            # Finalize current chunk
            add_chunk_to_results(current_chunk_elements)
            
            # Start new chunk
            current_chunk_elements = [element]
            current_chunk_text = text
            current_page = page_num
        else:
            # Add to current chunk
            if current_chunk_elements:
                current_chunk_text += " " + text
            else:
                current_chunk_text = text
            
            current_chunk_elements.append(element)
            current_page = page_num
    
    # The last chunk
    if current_chunk_elements:
        add_chunk_to_results(current_chunk_elements)
    
    return chunks


def _bbox(coords) -> Tuple[float, float, float, float]:
    """
    Return (xmin, ymin, xmax, ymax) for a CoordinatesFromTopLeft instance.
    Assumes each point has .x and .y attributes.
    """
    xs = [coords.top_left.x, coords.top_right.x,
          coords.bottom_left.x, coords.bottom_right.x]
    ys = [coords.top_left.y, coords.top_right.y,
          coords.bottom_left.y, coords.bottom_right.y]
    return min(xs), min(ys), max(xs), max(ys)


def _inside(inner, outer) -> bool:
    """
    True if *inner* bbox is entirely inside *outer* bbox.
    BBoxes are (xmin, ymin, xmax, ymax).
    """
    xi1, yi1, xi2, yi2 = inner
    xo1, yo1, xo2, yo2 = outer
    return xi1 >= xo1 and yi1 >= yo1 and xi2 <= xo2 and yi2 <= yo2


# ─── Main filter ────────────────────────────────────────────────────────────────
def remove_nested(elements: List[List[Any]]) -> List[List[Any]]:
    """
    elements: [file_type:str, file_data:base64, coords:CoordinatesFromTopLeft, page:int]

    Returns a **new** list where, for each page, any element whose rectangle is
    fully contained in another element’s rectangle is dropped.

    The biggest rectangle wins (we keep larger areas).
    """
    by_page = defaultdict(list)
    for el in elements:
        by_page[el[3]].append(el)            # group by page number

    filtered = []
    for page, items in by_page.items():
        # Build (element, bbox, area) triples
        triples = []
        for el in items:
            bb = _bbox(el[2])
            area = (bb[2] - bb[0]) * (bb[3] - bb[1])
            triples.append((el, bb, area))

        # Sort descending by area so we compare small against big
        triples.sort(key=lambda t: t[2], reverse=True)

        kept: List[Tuple[List[Any], Tuple[float, float, float, float]]] = []
        for el, bb, _ in triples:
            # If bb is inside any already-kept bbox, skip
            if any(_inside(bb, kept_bb) for _, kept_bb in kept):
                continue
            kept.append((el, bb))

        # Keep only the element part
        filtered.extend(el for el, _ in kept)

    return filtered

def extract_unknowns_as_images(
    texts: List[List[Any]],
    pdf_path: str,
) -> Tuple[List[List[Any]], List[List[Any]]]:
    """
    Walk *texts*.  Whenever the **second** field (text value) is '<unknown>',
    drop that entry from `clean_texts` and add an image stub that covers the
    whole page to `images_to_add`.

    Returns  (clean_texts, images_to_add)
    """
    doc            = fitz.open(pdf_path)
    clean_texts    = []
    images_to_add  = []

    for item in texts:
        if len(item) < 4:
            clean_texts.append(item)
            continue

        text_val  = item[1]       # first field is always "Text"
        page_num  = item[3]       # page number (1-based)

        if text_val == "<unknown>":
            page = doc.load_page(page_num - 1)
            r    = page.rect
            
            top_left = Point(x=r.x0, y=r.y0)
            top_right    = Point(x=r.x1, y=r.y0)
            bottom_left  = Point(x=r.x0, y=r.y1)
            bottom_right = Point(x=r.x1, y=r.y1)
            coords       = CoordinatesFromTopLeft(
                top_left=top_left,
                top_right=top_right,
                bottom_left=bottom_left,
                bottom_right=bottom_right,
                )


            images_to_add.append(
                ["Image", "uri_placeholder", coords, page_num]
            )
            
        else:
            clean_texts.append(item)
    return clean_texts, images_to_add