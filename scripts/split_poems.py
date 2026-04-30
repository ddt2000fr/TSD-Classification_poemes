import os
import re

def split_poems(input_file, output_dir, author, start_line, end_line):
    with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    
    lines = lines[start_line:end_line]
    
    poems = []
    current_poem = []
    current_title = None

    for line in lines:
        stripped = line.strip()
        
        if (stripped and
            stripped == stripped.upper() and
            len(stripped) > 2 and
            len(stripped) < 60 and
            re.search(r'[A-ZÀÂÄÉÈÊËÎÏÔÙÛÜÇ]', stripped)):
            
            if current_title and current_poem:
                text = ''.join(current_poem).strip()
                poem_lines = [l for l in current_poem if l.strip() and len(l.strip()) < 80]
                all_lines = [l for l in current_poem if l.strip()]
                if len(text) > 100 and len(all_lines) > 3 and len(poem_lines) / max(len(all_lines), 1) > 0.5:
                    poems.append((current_title, text))
            
            current_title = stripped
            current_poem = []
        else:
            if current_title:
                current_poem.append(line)
    
    if current_title and current_poem:
        text = ''.join(current_poem).strip()
        poem_lines = [l for l in current_poem if l.strip() and len(l.strip()) < 80]
        all_lines = [l for l in current_poem if l.strip()]
        if len(text) > 100 and len(all_lines) > 3 and len(poem_lines) / max(len(all_lines), 1) > 0.5:
            poems.append((current_title, text))
    
    os.makedirs(output_dir, exist_ok=True)
    for f in os.listdir(output_dir):
        if f.startswith('poeme_') and f.endswith('.txt'):
            os.remove(os.path.join(output_dir, f))
    
    for i, (title, text) in enumerate(poems):
        filename = f"{output_dir}/poeme_{i+1:03d}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"{title}\n\n{text}")
    
    print(f"{author}: {len(poems)} poèmes extraits")
    return len(poems)

split_poems('corpus/baudelaire/baudelaire_raw.txt',
            'corpus/baudelaire', 'Baudelaire', 303, 5009)

split_poems('corpus/verlaine/verlaine_raw.txt',
            'corpus/verlaine', 'Verlaine', 401, 9168)

split_poems('corpus/rimbaud/rimbaud_raw.txt',
            'corpus/rimbaud', 'Rimbaud', 353, 6761)

split_poems('corpus/gautier/gautier_raw.txt',
            'corpus/gautier', 'Gautier', 337, 10601)

split_poems('corpus/hugo/hugo_raw.txt',
            'corpus/hugo', 'Hugo', 174, 15157)
