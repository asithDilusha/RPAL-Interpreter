import argparse


from LexicalAnalyzer.lexer import tokenize as Lexer 
from Parser.parser import Parser

def main():
    print("RPAL Interpreter")

    parser = argparse.ArgumentParser(description='Process some RPAL files.')
    parser.add_argument('file_name', type=str, help='The RPAL program input file')
    parser.add_argument('-ast', action='store_true', help='Print the abstract syntax tree')
    parser.add_argument('-sast', action='store_true', help='Print the standardized abstract syntax tree')

    args = parser.parse_args()

    input_file = open(args.file_name, "r")
    input_text = input_file.read()
    input_file.close()
    
    # Tokenize the input text
    tokens = Lexer(input_text)
    print("Tokens:")

    for token in tokens:
        print(f"Type: {token.get_type()}, Value: {token.get_value()}")

    try:
        parser = Parser(tokens)
        ast_nodes = parser.parse()

        print("AST Nodes:")
        if ast_nodes is not None:
            for node in ast_nodes:
                print(node)
        else:
            print("No AST nodes generated.")


        if ast_nodes is None:
            return
        
        # Abstract Syntax Tree (AST)
        string_ast = parser.convert_ast_to_string_ast()
        if args.ast:
            for string in string_ast:
                print(string)
            return
        
        # Standardized Abstract Syntax Tree (SAST)
        #ast_factory = ASTFactory()
        #ast = ast_factory.get_abstract_syntax_tree(string_ast)
        #ast.standardize()
        #if args.sast:
            #ast.print_ast()
            #return
        
        # Final Output
        #cse_machine_factory = CSEMachineFactory()
        #cse_machine = cse_machine_factory.get_cse_machine(ast)
        
        # Default action: print the final output
        #print("Output of the above program is:")
        #print(cse_machine.get_answer())

    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
