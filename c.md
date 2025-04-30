#### lex-4
```
//lexp.l
%{
int COMMENT=0;
%}
identifier [a-zA-Z][a-zA-Z0-9]*
%%
#.* {printf ("\n %s is a Preprocessor Directive",yytext);}
int |
float |
main |
if |
else |
printf |
scanf |
for |
char |
getch |
while {printf("\n %s is a Keyword",yytext);}
"/*" {COMMENT=1;}
"*/" {COMMENT=0;}
{identifier}\( {if(!COMMENT) printf("\n Function:\t %s",yytext);}
\{ {if(!COMMENT) printf("\n Block Begins");
\} {if(!COMMENT) printf("\n Block Ends");}
{identifier}(\[[0-9]*\])? {if(!COMMENT) printf("\n %s is an Identifier",yytext);}
\".*\" {if(!COMMENT) printf("\n %s is a String",yytext);}
[0-9]+ {if(!COMMENT) printf("\n %s is a Number",yytext);}
\)(\;)? {if(!COMMENT) printf("\t");ECHO;printf("\n");}
\( ECHO;
= {if(!COMMENT) printf("\n%s is an Assmt oprtr",yytext);}
\<= |
\>= |
\< |
== {if(!COMMENT) printf("\n %s is a Rel. Operator",yytext);}
.|\n
%%
int main(int argc, char **argv)
{
if(argc>1)
{
FILE *file;
file=fopen(argv[1],"r");
if(!file)
{
printf("\n Could not open the file: %s",argv[1]);
exit(0);
}
yyin=file;
}
yylex();
printf("\n\n");
return 0;
}
int yywrap()
{
return 0;
}
//Output:
//test.c
#include<stdio.h>
main()
{
int fact=1,n;
for(int i=1;i<=n;i++)
{ fact=fact*i; }
printf("Factorial Value of N is", fact);
getch();
}
//$ lex lexp.l
//$ cc lex.yy.c
//$ ./a.out test.c
```
#### lex-5
```
%{
#include <stdio.h>
#include <stdlib.h>
#undef yywrap
#define yywrap() 1 
int f1 = 0, f2 = 0;
char oper;
float op1 = 0, op2 = 0, ans = 0;
void eval();
%}
DIGIT [0-9]
NUM {DIGIT}+(\.{DIGIT}+)?
OP [*/+-]
%%
{NUM} {
    if (f1 == 0) {
        op1 = atof(yytext);
        f1 = 1;
    } 
    else if (f2 == -1) {
        op2 = atof(yytext);
        f2 = 1;
    }
    if ((f1 == 1) && (f2 == 1)) {
        eval();
        f1 = 0;
        f2 = 0;
    }
}
{OP} {
    oper = yytext[0];
    f2 = -1;
}
[\n] {
    if (f1 == 1 && f2 == 1) {
        eval();
        f1 = 0;
        f2 = 0;
    }
}	
%%
int main() {
    printf("Enter an arithmetic expression:\n");
    yylex();
    return 0;
}
void eval() {
    switch (oper) {
        case '+':
            ans = op1 + op2;
            break;
        case '-':
            ans = op1 - op2;
            break;
        case '*':
            ans = op1 * op2;
            break;
        case '/':
            if (op2 == 0) {
                printf("ERROR: Division by zero\n");
                return;
            } else {
                ans = op1 / op2;
            }
            break;
        default:
            printf("ERROR: Invalid operator\n");
            return;
    }
    printf("The answer is: %f\n", ans);
}

```
#### SR
```
#include <bits/stdc++.h>
using namespace std;
int z = 0, i = 0, j = 0, c = 0;
char a[16], ac[20], stk[15], act[10];

void check() {
    strcpy(ac, "REDUCE TO E -> ");
    
    for(z = 0; z < c; z++) {
        if(stk[z] == '4') {
            printf("%s4", ac);
            stk[z] = 'E';
            stk[z + 1] = '\0';
            printf("\n$%s\t%s$\t", stk, a);
        }
    }
    for(z = 0; z < c - 2; z++) {
        if(stk[z] == '2' && stk[z + 1] == 'E' && stk[z + 2] == '2') {
            printf("%s2E2", ac);
            stk[z] = 'E';
            stk[z + 1] = '\0';
            stk[z + 2] = '\0';
            printf("\n$%s\t%s$\t", stk, a);
            i = i - 2;
        }
    }
    for(z = 0; z < c - 2; z++) {
        if(stk[z] == '3' && stk[z + 1] == 'E' && stk[z + 2] == '3') {
            printf("%s3E3", ac);
            stk[z] = 'E';
            stk[z + 1] = '\0';
            stk[z + 2] = '\0';
            printf("\n$%s\t%s$\t", stk, a);
            i = i - 2;
        }
    }
    return;
}

int main() {
    strcpy(a, "32423");
    c = strlen(a);
    strcpy(act, "SHIFT");
    
    printf("\nstack \t input \t action");
    printf("\n$\t%s$\t", a);

    for(i = 0; j < c; i++, j++) {
        printf("%s", act);
        stk[i] = a[j];
        stk[i + 1] = '\0';
        a[j] = ' ';
        printf("\n$%s\t%s$\t", stk, a);
        check();
    }
    check();
    
    if(stk[0] == 'E' && stk[1] == '\0')
        printf("Accept\n");
    else
        printf("Reject\n");
}
```

#### OP PRECEDENCE
```
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>  // For isdigit()

#define MAX 100

// --------------------------------------------------------------
// [1] : Operator Precedence Table
// --------------------------------------------------------------
// This table defines precedence relationships between operators.
// '>': Top of stack operator has higher precedence (Reduce)
// '<': Input operator has higher precedence (Shift)
// '=': Equal precedence (Shift)
// 'A': Accept (End of parsing)

char precedence_table[5][5] = {
    //  +    -    *    /    $
    {'>', '>', '<', '<', '>'}, // +
    {'>', '>', '<', '<', '>'}, // -
    {'>', '>', '>', '>', '>'}, // *
    {'>', '>', '>', '>', '>'}, // /
    {'<', '<', '<', '<', 'A'}  // $
};
char operators[] = "+-*/$";

// Returns the index of the operator in the precedence table.
// If the character is not an operator, returns -1.

int getIndex(char symbol) {
    for (int i = 0; i < 5; i++)
        if (operators[i] == symbol)
            return i;
    return -1;
}

// Defines a simple stack for parsing.
typedef struct {
    char stack[MAX];
    int top;
} Stack;

// Stack functions
void push(Stack *s, char symbol) {
    s->stack[++(s->top)] = symbol;
}
char pop(Stack *s) {
    return s->stack[(s->top)--];
}
char peek(Stack *s) {
    return s->stack[s->top];
}
int isOperator(char c) {
    return (c == '+' || c == '-' || c == '*' || c == '/' || c == '$');
}
void parseExpression(char *input) {
    Stack stack;
    stack.top = -1;
    push(&stack, '$'); // Push end marker
    int i = 0;

    printf("Stack\tInput\tAction\n");

    while (1) {
        // Skip spaces
        while (input[i] == ' ') i++;

        // Print stack content
        for (int j = 0; j <= stack.top; j++)
            printf("%c", stack.stack[j]);
        printf("\t%s\t", &input[i]);

        char top_symbol = peek(&stack);
        char next_symbol = input[i];

        // If the character is a number or variable, we shift.
        if (isdigit(next_symbol) || isalpha(next_symbol)) {
            printf("Shift (Operand: %c)\n", next_symbol);
            i++;
            while (isdigit(input[i]) || isalpha(input[i])) i++; // Skip full operand
            continue;
        }
        int top_index = getIndex(top_symbol);
        int next_index = getIndex(next_symbol);

        if (top_index == -1 || next_index == -1) {
            printf("Error: Invalid character in input\n");
            return;
        }
        char precedence = precedence_table[top_index][next_index];
        if (precedence == '<' || precedence == '=') {
            push(&stack, next_symbol);
            printf("Shift\n");
            i++;
        } 
        else if (precedence == '>') {
            while (getIndex(peek(&stack)) > getIndex(next_symbol)) {
                pop(&stack);
            }
            printf("Reduce\n");
        } 
        else if (precedence == 'A' && peek(&stack) == '$' && next_symbol == '$') {
            printf("Input accepted!\n");
            return;
        } 
        else {
            printf("Error: Invalid precedence relation\n");
            return;
        }
    }
}
int main() {
    char input[MAX];
    printf("Enter expression (end with $): ");
    fgets(input, MAX, stdin);  // Use fgets to handle spaces
    parseExpression(input);
    return 0;
```


#### LR
```
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

//---------------------------------------------------
// {1} : Stack Implementation for Parsing
//---------------------------------------------------

char stack[30];
int top = -1;

// Function to push a character onto the stack
void push(char c) {
    top++;
    stack[top] = c;
}

// Function to pop a character from the stack
char pop() {
    char c;
    if (top != -1) {
        c = stack[top];
        top--;
        return c;
    }
    return 'x'; // Return 'x' if the stack is empty
}

// Function to print the current status of the stack
void printstat() {
    int i;
    printf("\n\t\t\t $");
    for (i = 0; i <= top; i++)
        printf("%c", stack[i]);
}

//---------------------------------------------------
// {2} : Main Function to Simulate LR Parsing
//---------------------------------------------------

int main() {
    int i, l;
    char s1[20], ch1, ch2, ch3;

    // Prompt for input
    printf("\n\n\t\t LR PARSING");
    printf("\n\t\t ENTER THE EXPRESSION: ");
    scanf("%s", s1);
    l = strlen(s1);

    printf("\n\t\t $");

    //---------------------------------------------------
    // {3} : Token Processing Loop
    //      Convert "id" to 'E' and push operators
    //---------------------------------------------------

    for (i = 0; i < l; i++) {
        if (s1[i] == 'i' && s1[i + 1] == 'd') { // Detect 'id'
            s1[i] = ' ';         // Mark 'i' as processed
            s1[i + 1] = 'E';     // Replace 'd' with 'E' (non-terminal)
            
            printstat();
            printf(" id");

            push('E');           // Push non-terminal onto stack
            printstat();
        }
        else if (s1[i] == '+' || s1[i] == '-' || s1[i] == '*' || s1[i] == '/') {
            push(s1[i]);         // Push operator onto stack
            printstat();
        }
    }

    //---------------------------------------------------
    // {4} : Reduction Loop
    //      Reduce expressions in stack (E op E -> E)
    //---------------------------------------------------

    printstat();

    while (top >= 0) {
        ch1 = pop();             // Pop the top of the stack

        if (ch1 == 'x') {        // If stack is empty, break
            printf("\n\t\t\t $");
            break;
        }

        if (ch1 == '+' || ch1 == '/' || ch1 == '*' || ch1 == '-') {
            ch3 = pop();         // Expecting an 'E' after operator

            if (ch3 != 'E') {
                printf("error"); // Invalid reduction
                exit(1);
            }
            else {
                push('E');       // Successful reduction, push back 'E'
                printstat();
            }
        }

        ch2 = ch1;               // Store current character
    }
}
```
#### TAC
```
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

#define MAX 100

char stack[MAX];
int top = -1;
char tempVar = 't';

void push(char ch) {
    stack[++top] = ch;
}

char pop() {
    return stack[top--];
}

int precedence(char ch) {
    switch (ch) {
        case '+': case '-': return 1;
        case '*': case '/': return 2;
        case '^': return 3;
        default: return 0;
    }
}

void generateTAC(char *exp) {
    char output[MAX][MAX];
    int outputIndex = 0;
    char postfix[MAX];
    int postIndex = 0;
    
    for (int i = 0; exp[i] != '\0'; i++) {
        if (isalnum(exp[i])) {
            postfix[postIndex++] = exp[i];
        } else if (exp[i] == '(') {
            push(exp[i]);
        } else if (exp[i] == ')') {
            while (top != -1 && stack[top] != '(') {
                postfix[postIndex++] = pop();
            }
            pop(); 
        } else {
            while (top != -1 && precedence(stack[top]) >= precedence(exp[i])) {
                postfix[postIndex++] = pop();
            }
            push(exp[i]);
        }
    }
    
    while (top != -1) {
        postfix[postIndex++] = pop();
    }
    postfix[postIndex] = '\0';
    
    char operandStack[MAX][MAX];
    int operandTop = -1;
    tempVar = 't';
    
    for (int i = 0; postfix[i] != '\0'; i++) {
        if (isalnum(postfix[i])) {
            char operand[2] = {postfix[i], '\0'};
            strcpy(operandStack[++operandTop], operand);
        } else {
            char op2[MAX], op1[MAX];
            strcpy(op2, operandStack[operandTop--]);
            strcpy(op1, operandStack[operandTop--]);
            
            sprintf(output[outputIndex], "%c = %s %c %s", tempVar, op1, postfix[i], op2);
            sprintf(operandStack[++operandTop], "%c", tempVar);
            tempVar++;
            outputIndex++;
        }
    }
    
    printf("Three Address Code:\n");
    for (int i = 0; i < outputIndex; i++) {
        printf("%s\n", output[i]);
    }
}

int main() {
    char expression[MAX];
    printf("Enter an arithmetic expression: ");
    scanf("%s", expression);
    
    generateTAC(expression);
    
    return 0;
}

```
#### SYMBOL TABLE
```
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// -------------------------------------------------------
// [1] : Main Function - Builds a Symbol Table from Input Expression
// -------------------------------------------------------
int main() {
    int i = 0, j = 0, x = 0, n;
    void *p, *add[5];
    char ch, srch, b[15], d[15], c;

    printf("Expression terminated by $: ");

    // -------------------------------------------------------
    // [2] : Read Expression Until '$'
    // -------------------------------------------------------
    while ((c = getchar()) != '$') {
        b[i] = c;
        i++;
    }

    n = i - 1;

    // -------------------------------------------------------
    // [3] : Print the Given Expression
    // -------------------------------------------------------
    printf("Given Expression: ");
    i = 0;
    while (i <= n) {
        printf("%c", b[i]);
        i++;
    }

    // -------------------------------------------------------
    // [4] : Construct and Display Symbol Table
    // -------------------------------------------------------
    printf("\nSymbol Table\n");
    printf("Symbol \t addr \t type");

    while (j <= n) {
        c = b[j];

        if (isalpha(toascii(c))) {
            // Identifier
            p = malloc(sizeof(char));
            add[x] = p;
            d[x] = c;
            printf("\n%c \t %p \t identifier", c, p);
            x++;
        } else if (c == '+' || c == '-' || c == '*' || c == '=') {
            // Operator
            p = malloc(sizeof(char));
            add[x] = p;
            d[x] = c;
            printf("\n%c \t %p \t operator", c, p);
            x++;
        }
        j++;
    }

    return 0;
}
```

#### INTERMEDIATE CODE
```
//Input.txt
+ a b t1 
* c d t2 
- t1 t2 t
= t ? x

//Expt.c
#include <stdio.h>
#include <string.h>

char op[2], arg1[5], arg2[5], result[65];
void main() {
    FILE *fp1, *fp2;
    fp1 = fopen("/workspaces/codespaces-blank/input.txt", "r");
    if (fp1 == NULL) {
        perror("Error opening input file");
        return;
    }
    fp2 = fopen("/workspaces/codespaces-blank/output.txt", "w");
    if (fp2 == NULL) {
        perror("Error opening output file");
        fclose(fp1);
        return;
    }
    while (fscanf(fp1, "%s%s%s%s", op, arg1, arg2, result) == 4) {
        if (strcmp(op, "+") == 0) {
            fprintf(fp2, "\nMOV R0, %s", arg1);
            fprintf(fp2, "\nADD R0, %s", arg2);
            fprintf(fp2, "\nMOV %s, R0", result);
        }
        else if (strcmp(op, "*") == 0) {
            fprintf(fp2, "\nMOV R0, %s", arg1);
            fprintf(fp2, "\nMUL R0, %s", arg2);
            fprintf(fp2, "\nMOV %s, R0", result);
        }
        else if (strcmp(op, "-") == 0) {
            fprintf(fp2, "\nMOV R0, %s", arg1);
            fprintf(fp2, "\nSUB R0, %s", arg2);
            fprintf(fp2, "\nMOV %s, R0", result);
        }
        else if (strcmp(op, "/") == 0) {
            fprintf(fp2, "\nMOV R0, %s", arg1);
            fprintf(fp2, "\nDIV R0, %s", arg2);
            fprintf(fp2, "\nMOV %s, R0", result);
        }
        else if (strcmp(op, "=") == 0) {
            fprintf(fp2, "\nMOV R0, %s", arg1);
            fprintf(fp2, "\nMOV %s, R0", result);
        }
    }
    fclose(fp1);
    fclose(fp2);
    getchar(); 
}

//Output.txt
MOV R0, a
ADD R0, b
MOV t1, R0
MOV R0, c
MUL R0, d
MOV t2, R0
MOV R0, t1
SUB R0, t2
MOV t, R0
MOV R0, t
MOV x, R0

```

#### LEXICAL 
```
#include <iostream>
using namespace std;

int main() {
    int num = 10;
    int @value = 20; 
    // Lexical error: '@' is not a valid character in identifier names

    cout << num + @value << endl;
    return 0;
}
```

####  Syntax 
```
#include <iostream>
using namespace std;

int main() {
    int x = 10
    cout << "Value of x is: " << x << endl;
    // Missing semicolon ; after int x = 10 leads to a syntax error
    return 0;
}
```

#### Semantic 
```
#include <iostream>
using namespace std;

int main() {
    int totalApples = 10;
    int totalOranges = 5;

    int totalFruits = totalApples - totalOranges; 
// Semantic error: Wrong logic (should be addition)

    cout << "Total fruits: " << totalFruits << endl;
    return 0;
}
```
