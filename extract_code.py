import nbformat

def extract_code_from_notebook(notebook_path, output_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    with open(output_path, 'w', encoding='utf-8') as f:
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                f.write(cell['source'] + '\n\n')

# Example usage:
extract_code_from_notebook('fully_con_net_example.ipynb', 'fully_con_script.py')
