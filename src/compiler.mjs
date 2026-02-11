// guava-photon: Minimal AOT Concept Compiler
// TypeScript subset → WebAssembly (concept demonstration)
//
// This is NOT a production compiler. It demonstrates the concept of
// ahead-of-time compilation for a TypeScript subset, bypassing V8's JIT
// to achieve native-like execution speed.

import { readFileSync, writeFileSync } from 'fs';

// ============================================================
// Tokenizer (minimal)
// ============================================================

const TokenType = {
  NUMBER: 'NUMBER',
  IDENT: 'IDENT',
  PLUS: 'PLUS',
  MINUS: 'MINUS',
  STAR: 'STAR',
  SLASH: 'SLASH',
  LPAREN: 'LPAREN',
  RPAREN: 'RPAREN',
  LBRACE: 'LBRACE',
  RBRACE: 'RBRACE',
  COLON: 'COLON',
  SEMICOLON: 'SEMICOLON',
  COMMA: 'COMMA',
  LT: 'LT',
  LTEQ: 'LTEQ',
  GT: 'GT',
  GTEQ: 'GTEQ',
  EQ: 'EQ',
  EQEQ: 'EQEQ',
  RETURN: 'RETURN',
  FUNCTION: 'FUNCTION',
  IF: 'IF',
  ELSE: 'ELSE',
  CONST: 'CONST',
  LET: 'LET',
  EOF: 'EOF',
};

function tokenize(src) {
  const tokens = [];
  let i = 0;
  while (i < src.length) {
    if (/\s/.test(src[i])) { i++; continue; }
    if (src[i] === '/' && src[i+1] === '/') { while (i < src.length && src[i] !== '\n') i++; continue; }
    
    const simple = { '+': TokenType.PLUS, '-': TokenType.MINUS, '*': TokenType.STAR, '/': TokenType.SLASH,
      '(': TokenType.LPAREN, ')': TokenType.RPAREN, '{': TokenType.LBRACE, '}': TokenType.RBRACE,
      ':': TokenType.COLON, ';': TokenType.SEMICOLON, ',': TokenType.COMMA };
    
    if (simple[src[i]]) { tokens.push({ type: simple[src[i]] }); i++; continue; }
    
    if (src[i] === '<') { if (src[i+1] === '=') { tokens.push({ type: TokenType.LTEQ }); i += 2; } else { tokens.push({ type: TokenType.LT }); i++; } continue; }
    if (src[i] === '>') { if (src[i+1] === '=') { tokens.push({ type: TokenType.GTEQ }); i += 2; } else { tokens.push({ type: TokenType.GT }); i++; } continue; }
    if (src[i] === '=') { if (src[i+1] === '=') { tokens.push({ type: TokenType.EQEQ }); i += 2; } else { tokens.push({ type: TokenType.EQ }); i++; } continue; }
    
    if (/[0-9]/.test(src[i])) {
      let num = '';
      while (i < src.length && /[0-9.]/.test(src[i])) num += src[i++];
      tokens.push({ type: TokenType.NUMBER, value: parseFloat(num) });
      continue;
    }
    
    if (/[a-zA-Z_]/.test(src[i])) {
      let id = '';
      while (i < src.length && /[a-zA-Z0-9_]/.test(src[i])) id += src[i++];
      const keywords = { 'return': TokenType.RETURN, 'function': TokenType.FUNCTION, 'if': TokenType.IF, 'else': TokenType.ELSE, 'const': TokenType.CONST, 'let': TokenType.LET };
      tokens.push(keywords[id] ? { type: keywords[id] } : { type: TokenType.IDENT, value: id });
      continue;
    }
    i++;
  }
  tokens.push({ type: TokenType.EOF });
  return tokens;
}

// ============================================================
// Parser → AST (minimal recursive descent)
// ============================================================

function parse(tokens) {
  let pos = 0;
  const peek = () => tokens[pos];
  const eat = (t) => { if (tokens[pos].type !== t) throw new Error(`Expected ${t} got ${tokens[pos].type}`); return tokens[pos++]; };
  const match = (t) => { if (tokens[pos].type === t) return tokens[pos++]; return null; };

  function parseExpr() { return parseComparison(); }
  
  function parseComparison() {
    let left = parseAdditive();
    while ([TokenType.LT, TokenType.LTEQ, TokenType.GT, TokenType.GTEQ, TokenType.EQEQ].includes(peek().type)) {
      const op = tokens[pos++].type;
      const right = parseAdditive();
      left = { type: 'BinOp', op, left, right };
    }
    return left;
  }
  
  function parseAdditive() {
    let left = parseMultiplicative();
    while ([TokenType.PLUS, TokenType.MINUS].includes(peek().type)) {
      const op = tokens[pos++].type;
      left = { type: 'BinOp', op, left, right: parseMultiplicative() };
    }
    return left;
  }
  
  function parseMultiplicative() {
    let left = parsePrimary();
    while ([TokenType.STAR, TokenType.SLASH].includes(peek().type)) {
      const op = tokens[pos++].type;
      left = { type: 'BinOp', op, left, right: parsePrimary() };
    }
    return left;
  }
  
  function parsePrimary() {
    if (match(TokenType.LPAREN)) { const e = parseExpr(); eat(TokenType.RPAREN); return e; }
    if (peek().type === TokenType.NUMBER) return { type: 'Num', value: eat(TokenType.NUMBER).value };
    if (peek().type === TokenType.IDENT) {
      const name = eat(TokenType.IDENT).value;
      if (match(TokenType.LPAREN)) {
        const args = [];
        while (peek().type !== TokenType.RPAREN) { args.push(parseExpr()); match(TokenType.COMMA); }
        eat(TokenType.RPAREN);
        return { type: 'Call', name, args };
      }
      return { type: 'Var', name };
    }
    throw new Error(`Unexpected token: ${peek().type}`);
  }
  
  function parseStmt() {
    if (match(TokenType.RETURN)) { const e = parseExpr(); match(TokenType.SEMICOLON); return { type: 'Return', value: e }; }
    if (peek().type === TokenType.IF) return parseIf();
    if (match(TokenType.CONST) || match(TokenType.LET)) {
      const name = eat(TokenType.IDENT).value;
      if (match(TokenType.COLON)) { eat(TokenType.IDENT); } // skip type annotation
      eat(TokenType.EQ);
      const init = parseExpr();
      match(TokenType.SEMICOLON);
      return { type: 'VarDecl', name, init };
    }
    const e = parseExpr(); match(TokenType.SEMICOLON); return { type: 'ExprStmt', expr: e };
  }
  
  function parseIf() {
    eat(TokenType.IF);
    eat(TokenType.LPAREN);
    const cond = parseExpr();
    eat(TokenType.RPAREN);
    const then = parseBlock();
    let elseBlock = null;
    if (match(TokenType.ELSE)) elseBlock = parseBlock();
    return { type: 'If', cond, then, else: elseBlock };
  }
  
  function parseBlock() {
    eat(TokenType.LBRACE);
    const stmts = [];
    while (peek().type !== TokenType.RBRACE) stmts.push(parseStmt());
    eat(TokenType.RBRACE);
    return stmts;
  }
  
  function parseFunction() {
    eat(TokenType.FUNCTION);
    const name = eat(TokenType.IDENT).value;
    eat(TokenType.LPAREN);
    const params = [];
    while (peek().type !== TokenType.RPAREN) {
      const pName = eat(TokenType.IDENT).value;
      if (match(TokenType.COLON)) eat(TokenType.IDENT); // skip type
      params.push(pName);
      match(TokenType.COMMA);
    }
    eat(TokenType.RPAREN);
    if (match(TokenType.COLON)) eat(TokenType.IDENT); // skip return type
    const body = parseBlock();
    return { type: 'FuncDecl', name, params, body };
  }
  
  const program = [];
  while (peek().type !== TokenType.EOF) {
    if (peek().type === TokenType.FUNCTION) program.push(parseFunction());
    else program.push(parseStmt());
  }
  return program;
}

// ============================================================
// Codegen: AST → WebAssembly Text Format (WAT)
// ============================================================

function codegen(ast) {
  const functions = ast.filter(n => n.type === 'FuncDecl');
  let wat = '(module\n';
  
  for (const fn of functions) {
    const locals = new Map();
    fn.params.forEach((p, i) => locals.set(p, i));
    let localIdx = fn.params.length;
    
    function collectLocals(stmts) {
      for (const s of stmts) {
        if (s.type === 'VarDecl' && !locals.has(s.name)) { locals.set(s.name, localIdx++); }
        if (s.type === 'If') { collectLocals(s.then); if (s.else) collectLocals(s.else); }
      }
    }
    collectLocals(fn.body);
    
    const paramDecl = fn.params.map(p => `(param $${p} f64)`).join(' ');
    const extraLocals = [];
    locals.forEach((idx, name) => { if (idx >= fn.params.length) extraLocals.push(`(local $${name} f64)`); });
    
    wat += `  (func $${fn.name} ${paramDecl} (result f64)\n`;
    if (extraLocals.length) wat += `    ${extraLocals.join(' ')}\n`;
    
    function emitExpr(e) {
      if (e.type === 'Num') return `(f64.const ${e.value})`;
      if (e.type === 'Var') return `(local.get $${e.name})`;
      if (e.type === 'Call') return `(call $${e.name} ${e.args.map(emitExpr).join(' ')})`;
      if (e.type === 'BinOp') {
        const ops = { PLUS: 'f64.add', MINUS: 'f64.sub', STAR: 'f64.mul', SLASH: 'f64.div',
          LT: 'f64.lt', LTEQ: 'f64.le', GT: 'f64.gt', GTEQ: 'f64.ge', EQEQ: 'f64.eq' };
        const wasmOp = ops[e.op];
        const inner = `(${wasmOp} ${emitExpr(e.left)} ${emitExpr(e.right)})`;
        if (['LT','LTEQ','GT','GTEQ','EQEQ'].includes(e.op)) {
          return `(select (f64.const 1) (f64.const 0) ${inner})`;
        }
        return inner;
      }
      throw new Error(`Unknown expr: ${e.type}`);
    }
    
    function emitStmt(s) {
      if (s.type === 'Return') return `    (return ${emitExpr(s.value)})\n`;
      if (s.type === 'VarDecl') return `    (local.set $${s.name} ${emitExpr(s.init)})\n`;
      if (s.type === 'ExprStmt') return `    ${emitExpr(s.expr)}\n    drop\n`;
      if (s.type === 'If') {
        let code = `    (if (result f64) (i32.trunc_f64_s ${emitExpr(s.cond)})\n      (then\n`;
        s.then.forEach(st => code += '  ' + emitStmt(st));
        code += `        (f64.const 0)\n      )\n`;
        if (s.else) {
          code += `      (else\n`;
          s.else.forEach(st => code += '  ' + emitStmt(st));
          code += `        (f64.const 0)\n      )\n`;
        } else {
          code += `      (else (f64.const 0))\n`;
        }
        code += `    )\n    drop\n`;
        return code;
      }
      throw new Error(`Unknown stmt: ${s.type}`);
    }
    
    fn.body.forEach(s => { wat += emitStmt(s); });
    wat += `    (f64.const 0)\n  )\n`;
    wat += `  (export "${fn.name}" (func $${fn.name}))\n`;
  }
  
  wat += ')\n';
  return wat;
}

// ============================================================
// Main
// ============================================================

const input = process.argv[2];
if (!input) {
  console.log('guava-photon AOT Compiler (concept)');
  console.log('Usage: node compiler.mjs <input.ts>');
  console.log('');
  console.log('Compiles a TypeScript subset to WebAssembly Text Format (WAT).');
  console.log('Demonstrates AOT compilation without V8 JIT overhead.');
  process.exit(0);
}

const src = readFileSync(input, 'utf-8');
const tokens = tokenize(src);
const ast = parse(tokens);
const wat = codegen(ast);

const outFile = input.replace(/\.(ts|js|mjs)$/, '.wat');
writeFileSync(outFile, wat);
console.log(`✅ Compiled ${input} → ${outFile}`);
console.log(`   Functions: ${ast.filter(n => n.type === 'FuncDecl').length}`);
console.log(`   WAT size: ${wat.length} bytes`);
