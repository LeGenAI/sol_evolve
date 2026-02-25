import re
import glob
import os

# Exceptions for acronyms or proper nouns to keep capitalized
KEEP_CAPS = [
    "SAT", "LLM", "SolEvolve", "AI", "BKLC", "GA", "BCH", "Grassl", 
    "Mollard", "CNF", "Gurobi", "Magma", "MAC", "AWGN", "Boolean", 
    "Lucas", "ILP", "UNSAT", "SAT-GA", "OpenRouter", "GitHub", "CSAT",
    "Tseitin"
]

def sentence_case(text):
    # If the text is empty or too short, return as is
    if not text.strip():
        return text
    
    # Capitalize the first letter (even if it's inside a command, try to do it right, but usually headings start with text)
    # Actually just lowercasing everything and then fixing the first letter and exceptions is easier.
    # But wait, LaTeX math $...$ shouldn't be lowercased.
    # To be safe, let's keep it simple: split by spaces, lowercase mostly.
    
    # A better approach: 
    # 1. Temporarily extract math $...$
    math_blocks = []
    def repl_math(m):
        math_blocks.append(m.group(0))
        return f"__MATH_{len(math_blocks)-1}__"
    
    text_no_math = re.sub(r'\$.*?\$', repl_math, text)
    
    # 2. Lowercase everything except first character
    first_char_idx = next((i for i, c in enumerate(text_no_math) if c.isalpha()), -1)
    if first_char_idx != -1:
        rest = text_no_math[first_char_idx+1:].lower()
        text_no_math = text_no_math[:first_char_idx] + text_no_math[first_char_idx].upper() + rest
    else:
        text_no_math = text_no_math.lower()
        
    # 3. Restore proper nouns
    for word in KEEP_CAPS:
        # Case insensitive replace to restore caps
        text_no_math = re.sub(rf'\b{word}\b', word, text_no_math, flags=re.IGNORECASE)
        # also handle cases attached to hyphens like SAT-based -> SAT-based
        
    # 4. Restore math
    def repl_restore(m):
        idx = int(m.group(1))
        return math_blocks[idx]
    
    result = re.sub(r'__MATH_(\d+)__', repl_restore, text_no_math)
    return result

def process_headings(content):
    def repl_heading(m):
        cmd = m.group(1) # e.g. section, subsection*, paragraph
        title = m.group(2)
        new_title = sentence_case(title)
        return f"\\{cmd}{{{new_title}}}"
        
    # Match \section{...}, \subsection*{...}, \paragraph{...}
    # Handling nested braces perfectly is hard in regex, but usually there are no nested braces in our headings,
    # or at most one level.
    # Regex for one level of nested braces: \{([^{}]*|{[^{}]*})\}
    pattern = re.compile(r'\\((?:sub)*section\*?|paragraph\*?)\{([^{}]*)\}')
    return pattern.sub(repl_heading, content)

def process_code_mentions(content):
    # We want to ensure that things like [32,14,8] or [22,11] become "binary [32,14,8] code"
    # But ONLY in text, not in math equations. And we don't want "binary binary" or "code code".
    # And we want to leave alone things like "[1]" (citations) or equation references.
    # A code parameter is typically [n,k] or [n,k,d] where n>k.
    
    # Find patterns like (optional "an " or "a ") (optional "binary ") \$?\[\d+,\d+(?:,\d+)?\]\$? (optional " code")
    # Actually, the user says "항상 코드를 명시할 때에는 a binary code 라는 것을 넣어야 해!"
    # So "the [32,14,8]" -> "the binary [32,14,8] code".
    # "an [n,k,d]" -> "a binary [n,k,d] code"
    
    def repl_code(m):
        prefix = m.group(1) or ""
        binary_word = m.group(2) or ""
        bracket = m.group(3) # e.g. 32,14,8
        suffix = m.group(4) or ""
        
        # Determine if it's really a code parameter: usually at least two numbers 
        if "," not in bracket: 
            return m.group(0) # Not a code parameter (e.g., citation [1])
            
        # Fix "an " -> "a " if we prepend "binary"
        if prefix.lower().strip() == "an":
            prefix = prefix.replace("an", "a").replace("An", "A")
            
        # Build the replacement string
        # If there's no "binary " before it, add it.
        pre = prefix
        if not re.search(r'(?i)\bbinary\s+', pre):
            pre += "binary "
            
        # Ensure bracket is in math mode
        core = f"$[{bracket}]$"
        
        # Ensure it is followed by "code" or "codes"
        post = suffix
        if not re.search(r'(?i)\bcodes?\b', post):
            post = " code" + post
            
        return pre + core + post

    # Matches:
    # (word before like 'a ', 'an ', 'the ', 'optimal ', 'inequivalent ', 'binary ')? 
    # \$?\[(\d+,\s*\d+(?:,\s*\d+)?)\]\$?
    # (\s+code(?:s)?)?
    
    # Let's use a simpler pattern and just fix occurrences.
    # Pattern: match optional "an ", "a ", "the " etc.
    # Actually, simpler:
    # (an?\s+|the\s+)?(binary\s+)?\$?\[(\d+,\s*\d+(?:,\s*\d+)?)\]\$?(?:\s*code(?:s)?)?
    
    pattern = re.compile(r'\b(an?\s+|the\s+|optimal\s+|inequivalent\s+)?(binary\s+)?\$?\[(\d+,\s*\d+(?:,\s*\d+)?)\]\$?(\s+codes?)?', re.IGNORECASE)
    
    return pattern.sub(repl_code, content)

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    new_content = process_headings(content)
    # The code mentions replacement is very sensitive. Let's do a more careful replace.
    # Let's write a customized function for the second rule:
    
    def safe_code_repl(m):
        full_match = m.group(0)
        prefix = m.group(1) or ""
        params = m.group(2)
        suffix = m.group(3) or ""
        
        # Don't touch if it looks like a citation or non-code
        if "," not in params:
            return full_match
            
        # Check if "binary" is already in prefix or nearby. 
        # Actually user said "항상 코드를 명시할 때에는 a binary code 라"
        # We'll just enforce "binary $[params]$ code" if "binary" or "code" is missing.
        
        # If prefix is "an ", make it "a "
        if prefix.lower().endswith("an "):
            prefix = prefix[:-3] + ("A " if prefix.endswith("An ") else "a ")
            
        # Add "binary" if missing
        if "binary" not in prefix.lower():
            if prefix and not prefix.endswith(" "):
                prefix += " "
            prefix += "binary "
            
        # Add "code" if missing
        if "code" not in suffix.lower():
            suffix = " code" + suffix
            
        return f"{prefix}$[{params}]${suffix}"
        
    # Match things like: [32,14,8] or $[32,14,8]$ 
    # with up to 2 words before and 1 word after to check for 'binary' and 'code'
    # Actually, a simpler regex is just matching the brackets, then we look around.
    
    # We'll do a simple regex and manually adjust
    # This regex catches $[n,k,d]$ or [n,k,d]
    code_pattern = re.compile(r'(?:\b(?:an?|the|optimal|inequivalent|binary)\s+)*\$?\[(\d+,\s*\d+(?:,\s*\d+)?)\]\$?(?:\s+codes?)?', re.IGNORECASE)
    
    # Just use the repl_code we wrote above, it is good enough
    new_content = process_code_mentions(new_content)
    
    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {filepath}")

def main():
    tex_files = glob.glob('/Users/baegjaehyeon/Desktop/ICML/icml_2026_solevolve/els-cas-templates/**/*.tex', recursive=True)
    tex_files += glob.glob('/Users/baegjaehyeon/Desktop/ICML/icml_2026_solevolve/els-cas-templates/*.tex')
    
    for tf in set(tex_files):
        # don't modify cas-dc-template.tex or other style files if they exist, only our sections
        if "cas-" in os.path.basename(tf) and "template" in tf:
            continue
        process_file(tf)

if __name__ == "__main__":
    main()
