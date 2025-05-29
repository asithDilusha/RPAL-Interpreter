import sys
from enum import Enum
from LexicalAnalyzer.lexer import MyToken, TokenType
from typing import List, Optional

class NodeType(Enum):
    let = 1
    fcn_form = 2
    identifier = 3
    integer = 4
    string = 5
    where = 6
    gamma = 7
    lambda_expr = 8
    tau = 9
    rec = 10
    aug = 11
    conditional = 12
    op_or = 13
    op_and = 14
    op_not = 15
    op_compare = 16
    op_plus = 17
    op_minus = 18
    op_neg = 19
    op_mul = 20
    op_div = 21
    op_pow = 22
    at = 23
    true_value = 24
    false_value = 25
    nil = 26
    dummy = 27
    within = 28
    and_op = 29
    equal = 30
    comma = 31
    empty_params = 32

class Node:
    def __init__(self, node_type: NodeType, value: str, children: int):
        self.type = node_type
        self.value = value
        self.no_of_children = children

class Parser:
    def __init__(self, tokens: List[MyToken]):
        self.tokens = tokens
        self.ast: List[Node] = []
        self.string_ast: List[str] = []
        self.current_token_index = 0

    def parse(self) -> Optional[List[Node]]:
        self.tokens.append(MyToken(TokenType.END_OF_TOKENS, ""))
        self.E()
        if self.tokens[0].type == TokenType.END_OF_TOKENS:
            return self.ast
        else:
            print("Parsing Unsuccessful!...........")
            print("REMAINIG UNPARSED TOKENS:")
            for token in self.tokens:
                print("<" + str(token.type) + ", " + token.value + ">")
            return None

    def convert_ast_to_string_ast(self) -> List[str]:
        dots = ""
        stack = []

        while self.ast:
            if not stack:
                if self.ast[-1].no_of_children == 0:
                    self.add_strings(dots, self.ast.pop())
                else:
                    node = self.ast.pop()
                    stack.append(node)
            else:
                if self.ast[-1].no_of_children > 0:
                    node = self.ast.pop()
                    stack.append(node)
                    dots += "."
                else:
                    stack.append(self.ast.pop())
                    dots += "."
                    while stack[-1].no_of_children == 0:
                        self.add_strings(dots, stack.pop())
                        if not stack:
                            break
                        dots = dots[:-1]
                        node = stack.pop()
                        node.no_of_children -= 1
                        stack.append(node)

        self.string_ast.reverse()
        return self.string_ast

    def add_strings(self, dots: str, node: Node) -> None:
        if node.type in [NodeType.identifier, NodeType.integer, NodeType.string, NodeType.true_value,
                         NodeType.false_value, NodeType.nil, NodeType.dummy]:
            self.string_ast.append(dots + "<" + node.type.name.upper() + ":" + node.value + ">")
        elif node.type == NodeType.fcn_form:
            self.string_ast.append(dots + "function_form")
        else:
            self.string_ast.append(dots + node.value)

    def consume_token(self) -> Optional[MyToken]:
        """Safely consume a token from the list."""
        if not self.tokens:
            return None
        token = self.tokens.pop(0)
        return token

    def peek_token(self) -> Optional[MyToken]:
        """Safely peek at the next token."""
        return self.tokens[0] if self.tokens else None

    def expect_token(self, expected_type: TokenType, expected_value: Optional[str] = None) -> bool:
        """Check if the next token matches the expected type and value."""
        if not self.tokens:
            return False
        token = self.tokens[0]
        return (token.type == expected_type and 
                (expected_value is None or token.value == expected_value))

    def E(self) -> None:
        if not self.tokens:
            return

        token = self.peek_token()
        if not token:
            return

        if token.type == TokenType.KEYWORD and token.value in ["let", "fn"]:
            if token.value == "let":
                self.consume_token()  # Remove "let"
                self.D()
                if not self.expect_token(TokenType.KEYWORD, "in"):
                    print("Parse error at E : 'in' Expected")
                    return
                self.consume_token()  # Remove "in"
                self.E()
                self.ast.append(Node(NodeType.let, "let", 2))
            else:
                self.consume_token()  # Remove "fn"
                n = 0
                while self.tokens and (self.tokens[0].type == TokenType.IDENTIFIER or self.tokens[0].value == "("):
                    self.Vb()
                    n += 1
                if not self.expect_token(TokenType.OPERATOR, "."):
                    print("Parse error at E : '.' Expected")
                    return
                self.consume_token()  # Remove "."
                self.E()
                self.ast.append(Node(NodeType.lambda_expr, "lambda", n + 1))
        else:
            self.Ew()

    def Ew(self) -> None:
        self.T()
        if self.expect_token(TokenType.KEYWORD, "where"):
            self.consume_token()  # Remove "where"
            self.Dr()
            self.ast.append(Node(NodeType.where, "where", 2))

    def T(self) -> None:
        self.Ta()
        n = 1
        while self.expect_token(TokenType.PUNCTUATION, ","):
            self.consume_token()  # Remove comma
            self.Ta()
            n += 1
        if n > 1:
            self.ast.append(Node(NodeType.tau, "tau", n))

    def Ta(self) -> None:
        self.Tc()
        while self.expect_token(TokenType.KEYWORD, "aug"):
            self.consume_token()  # Remove "aug"
            self.Tc()
            self.ast.append(Node(NodeType.aug, "aug", 2))

    def Tc(self) -> None:
        self.B()
        if self.expect_token(TokenType.OPERATOR, "->"):
            self.consume_token()  # Remove '->'
            self.Tc()
            if not self.expect_token(TokenType.OPERATOR, "|"):
                print("Parse error at Tc: conditional '|' expected")
                return
            self.consume_token()  # Remove '|'
            self.Tc()
            self.ast.append(Node(NodeType.conditional, "->", 3))

    def B(self) -> None:
        self.Bt()
        while self.expect_token(TokenType.KEYWORD, "or"):
            self.consume_token()  # Remove 'or'
            self.Bt()
            self.ast.append(Node(NodeType.op_or, "or", 2))

    def Bt(self) -> None:
        self.Bs()
        while self.expect_token(TokenType.OPERATOR, "&"):
            self.consume_token()  # Remove '&'
            self.Bs()
            self.ast.append(Node(NodeType.op_and, "&", 2))

    def Bs(self) -> None:
        if self.expect_token(TokenType.KEYWORD, "not"):
            self.consume_token()  # Remove 'not'
            self.Bp()
            self.ast.append(Node(NodeType.op_not, "not", 1))
        else:
            self.Bp()

    def Bp(self) -> None:
        self.A()
        token = self.peek_token()
        if token and token.value in [">", ">=", "<", "<=", "gr", "ge", "ls", "le", "eq", "ne"]:
            self.consume_token()
            self.A()
            if token.value == ">":
                self.ast.append(Node(NodeType.op_compare, "gr", 2))
            elif token.value == ">=":
                self.ast.append(Node(NodeType.op_compare, "ge", 2))
            elif token.value == "<":
                self.ast.append(Node(NodeType.op_compare, "ls", 2))
            elif token.value == "<=":
                self.ast.append(Node(NodeType.op_compare, "le", 2))
            else:
                self.ast.append(Node(NodeType.op_compare, token.value, 2))

    def A(self) -> None:
        if self.expect_token(TokenType.OPERATOR, "+"):
            self.consume_token()  # Remove unary plus
            self.At()
        elif self.expect_token(TokenType.OPERATOR, "-"):
            self.consume_token()  # Remove unary minus
            self.At()
            self.ast.append(Node(NodeType.op_neg, "neg", 1))
        else:
            self.At()

        while self.peek_token() and self.peek_token().value in {"+", "-"}:
            current_token = self.consume_token()
            self.At()
            if current_token.value == "+":
                self.ast.append(Node(NodeType.op_plus, "+", 2))
            else:
                self.ast.append(Node(NodeType.op_minus, "-", 2))

    def At(self) -> None:
        self.Af()
        while self.peek_token() and self.peek_token().value in {"*", "/"}:
            current_token = self.consume_token()
            self.Af()
            if current_token.value == "*":
                self.ast.append(Node(NodeType.op_mul, "*", 2))
            else:
                self.ast.append(Node(NodeType.op_div, "/", 2))

    def Af(self) -> None:
        self.Ap()
        if self.expect_token(TokenType.OPERATOR, "**"):
            self.consume_token()  # Remove power operator
            self.Af()
            self.ast.append(Node(NodeType.op_pow, "**", 2))

    def Ap(self) -> None:
        self.R()
        while self.expect_token(TokenType.OPERATOR, "@"):
            self.consume_token()  # Remove @ operator
            if not self.expect_token(TokenType.IDENTIFIER):
                print("Parsing error at Ap: IDENTIFIER EXPECTED")
                return
            self.ast.append(Node(NodeType.identifier, self.peek_token().value, 0))
            self.consume_token()  # Remove IDENTIFIER
            self.R()
            self.ast.append(Node(NodeType.at, "@", 3))

    def R(self) -> None:
        self.Rn()
        while (self.peek_token() and 
               (self.peek_token().type in [TokenType.IDENTIFIER, TokenType.INTEGER, TokenType.STRING] or
                self.peek_token().value in ["true", "false", "nil", "dummy"] or
                self.peek_token().value == "(")):
            self.Rn()
            self.ast.append(Node(NodeType.gamma, "gamma", 2))

    def Rn(self) -> None:
        token = self.peek_token()
        if not token:
            return

        if token.type == TokenType.IDENTIFIER:
            self.ast.append(Node(NodeType.identifier, token.value, 0))
            self.consume_token()
        elif token.type == TokenType.INTEGER:
            self.ast.append(Node(NodeType.integer, token.value, 0))
            self.consume_token()
        elif token.type == TokenType.STRING:
            self.ast.append(Node(NodeType.string, token.value, 0))
            self.consume_token()
        elif token.type == TokenType.KEYWORD:
            if token.value == "true":
                self.ast.append(Node(NodeType.true_value, token.value, 0))
                self.consume_token()
            elif token.value == "false":
                self.ast.append(Node(NodeType.false_value, token.value, 0))
                self.consume_token()
            elif token.value == "nil":
                self.ast.append(Node(NodeType.nil, token.value, 0))
                self.consume_token()
            elif token.value == "dummy":
                self.ast.append(Node(NodeType.dummy, token.value, 0))
                self.consume_token()
            else:
                print("Parse Error at Rn: Unexpected KEYWORD")
        elif token.type == TokenType.PUNCTUATION:
            if token.value == "(":
                self.consume_token()  # Remove '('
                self.E()
                if not self.expect_token(TokenType.PUNCTUATION, ")"):
                    print("Parsing error at Rn: Expected a matching ')'")
                    return
                self.consume_token()  # Remove ')'
            else:
                print("Parsing error at Rn: Unexpected PUNCTUATION")
        else:
            print(token.type, token.value)
            print("Parsing error at Rn: Expected a Rn, but got different")

    def D(self) -> None:
        self.Da()
        if self.expect_token(TokenType.KEYWORD, "within"):
            self.consume_token()  # Remove 'within'
            self.D()
            self.ast.append(Node(NodeType.within, "within", 2))

    def Da(self) -> None:
        self.Dr()
        n = 1
        while self.expect_token(TokenType.KEYWORD, "and"):
            self.consume_token()
            self.Dr()
            n += 1
        if n > 1:
            self.ast.append(Node(NodeType.and_op, "and", n))

    def Dr(self) -> None:
        is_rec = False
        if self.expect_token(TokenType.KEYWORD, "rec"):
            self.consume_token()
            is_rec = True
        self.Db()
        if is_rec:
            self.ast.append(Node(NodeType.rec, "rec", 1))

    def Db(self) -> None:
        if self.expect_token(TokenType.PUNCTUATION, "("):
            self.consume_token()
            self.D()
            if not self.expect_token(TokenType.PUNCTUATION, ")"):
                print("Parsing error at Db #1")
                return
            self.consume_token()
        elif self.expect_token(TokenType.IDENTIFIER):
            if len(self.tokens) > 1 and (self.tokens[1].value == "(" or self.tokens[1].type == TokenType.IDENTIFIER):
                self.ast.append(Node(NodeType.identifier, self.peek_token().value, 0))
                self.consume_token()  # Remove ID
                n = 1  # Identifier child
                while self.peek_token() and (self.peek_token().type == TokenType.IDENTIFIER or self.peek_token().value == "("):
                    self.Vb()
                    n += 1
                if not self.expect_token(TokenType.OPERATOR, "="):
                    print("Parsing error at Db #2")
                    return
                self.consume_token()
                self.E()
                self.ast.append(Node(NodeType.fcn_form, "fcn_form", n+1))
            elif len(self.tokens) > 1 and self.tokens[1].value == "=":
                self.ast.append(Node(NodeType.identifier, self.peek_token().value, 0))
                self.consume_token()  # Remove identifier
                self.consume_token()  # Remove equal
                self.E()
                self.ast.append(Node(NodeType.equal, "=", 2))
            elif len(self.tokens) > 1 and self.tokens[1].value == ",":
                self.Vl()
                if not self.expect_token(TokenType.OPERATOR, "="):
                    print("Parsing error at Db")
                    return
                self.consume_token()
                self.E()
                self.ast.append(Node(NodeType.equal, "=", 2))

    def Vb(self) -> None:
        if self.expect_token(TokenType.PUNCTUATION, "("):
            self.consume_token()
            isVl = False
            if self.expect_token(TokenType.IDENTIFIER):
                self.Vl()
                isVl = True
            if not self.expect_token(TokenType.PUNCTUATION, ")"):
                print("Parse error unmatch )")
                return
            self.consume_token()
            if not isVl:
                self.ast.append(Node(NodeType.empty_params, "()", 0))
        elif self.expect_token(TokenType.IDENTIFIER):
            self.ast.append(Node(NodeType.identifier, self.peek_token().value, 0))
            self.consume_token()

    def Vl(self) -> None:
        n = 0
        while True:
            if n > 0:
                self.consume_token()
            if not self.expect_token(TokenType.IDENTIFIER):
                print("Parse error: an identifier was expected")
                return
            self.ast.append(Node(NodeType.identifier, self.peek_token().value, 0))
            self.consume_token()
            n += 1
            if not self.expect_token(TokenType.PUNCTUATION, ","):
                break
        if n > 1:
            self.ast.append(Node(NodeType.comma, ",", n))

def parse_program(input_str):
    """Parse a program string and return the AST."""
    from LexicalAnalyzer.lexical_analyzer import tokenize
    tokens = tokenize(input_str)
    parser = Parser(tokens)
    ast = parser.parse()
    return parser

def print_ast(parser):
    """Print the AST in the format used by parser_1.py."""
    string_ast = parser.convert_ast_to_string_ast()
    for line in string_ast:
        print(line)

def main():
    if len(sys.argv) != 2:
        print("Usage: python parser_adapted.py <input_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    try:
        with open(input_file, 'r') as f:
            input_str = f.read()
        
        parser = parse_program(input_str)
        print_ast(parser)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
