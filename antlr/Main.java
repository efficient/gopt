import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.LinkedList;
import java.util.Scanner;

import org.antlr.v4.runtime.ANTLRInputStream;
import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.ParserRuleContext;
import org.antlr.v4.runtime.TokenStreamRewriter;
import org.antlr.v4.runtime.tree.ParseTreeWalker;

public class Main {
	static String gotoFilePath = "/Users/akalia/Documents/workspace/fastpp/test/nogoto.c";
	
	public static void main(String args[]) throws FileNotFoundException {
		String code = getCode(gotoFilePath);
	
		LinkedList<VariableDecl> localVars = extractLocalVariables(code);
		code = trimDeclarations(code);
		code = cleanup(code);
		code = vectorizeLocalVariables(code, localVars);
	
		// This should be done after vectorizing local variable usage
		code = insertLocalVariableDeclarations(code, localVars);
		code = deleteForeach(code, localVars);
		
		System.out.flush();
		System.err.println("\nFinal code:");
		System.err.flush();
		System.out.println(code);
		
		PrintWriter out = new PrintWriter(new File(gotoFilePath + ".proc"));
		out.println(code);
		out.close();
	}
	
	private static String deleteForeach(String code,
			LinkedList<VariableDecl> localVars) {
		System.out.println("\n\n Deleting foreach");

		CharStream charStream = new ANTLRInputStream(code);		
		CLexer lexer = new CLexer(charStream);
		CommonTokenStream tokens = new CommonTokenStream(lexer);
		TokenStreamRewriter rewriter = new TokenStreamRewriter(tokens);
		CParser parser = new CParser(tokens);
		
		// Parse and get the root of the parse tree
		ParserRuleContext tree = parser.compilationUnit();

		ForeachDeleter feDeleter = new ForeachDeleter(parser, rewriter);
		
		ParseTreeWalker walker = new ParseTreeWalker();
		walker.walk(feDeleter, tree);
		
		return rewriter.getText();
	}

	private static String insertLocalVariableDeclarations(String code,
			LinkedList<VariableDecl> localVars) {
		System.out.println("\n\nInserting local variable declarations");

		CharStream charStream = new ANTLRInputStream(code);		
		CLexer lexer = new CLexer(charStream);
		CommonTokenStream tokens = new CommonTokenStream(lexer);
		TokenStreamRewriter rewriter = new TokenStreamRewriter(tokens);
		CParser parser = new CParser(tokens);
		
		// Parse and get the root of the parse tree
		ParserRuleContext tree = parser.compilationUnit();

		DeclarationInserter dInserter = new DeclarationInserter(parser, 
				rewriter, localVars);
		
		ParseTreeWalker walker = new ParseTreeWalker();
		walker.walk(dInserter, tree);
		
		System.err.println();  // We print the replaced local vars on a single line
		return rewriter.getText();
	}

	private static String vectorizeLocalVariables(String code, 
			LinkedList<VariableDecl> localVars) {
		System.out.println("\n\nVectorizing local variables");

		CharStream charStream = new ANTLRInputStream(code);		
		CLexer lexer = new CLexer(charStream);
		CommonTokenStream tokens = new CommonTokenStream(lexer);
		TokenStreamRewriter rewriter = new TokenStreamRewriter(tokens);
		CParser parser = new CParser(tokens);
		
		// Parse and get the root of the parse tree
		ParserRuleContext tree = parser.compilationUnit();

		LocalVariableVectorizer lvVectorizer = new LocalVariableVectorizer(parser, 
				rewriter, localVars);
		
		ParseTreeWalker walker = new ParseTreeWalker();
		walker.walk(lvVectorizer, tree);
		
		System.err.println();  // We print the replaced local vars on a single line
		return rewriter.getText();
	}


	private static String cleanup(String code) {
		System.out.println("\n\nRunning cleanup");
		
		CharStream charStream = new ANTLRInputStream(code);
		CLexer lexer = new CLexer(charStream);
		CommonTokenStream tokens = new CommonTokenStream(lexer);
		TokenStreamRewriter rewriter = new TokenStreamRewriter(tokens);
		CParser parser = new CParser(tokens);
		
		// Parse and get the root of the parse tree
		ParserRuleContext tree = parser.compilationUnit();

		CodeCleaner cleaner = new CodeCleaner(parser, rewriter);
		
		ParseTreeWalker walker = new ParseTreeWalker();
		walker.walk(cleaner, tree);
		
		return rewriter.getText();

	}

	private static String trimDeclarations(String code) {
		System.out.println("\n\nTrimming declarations");

		CharStream charStream = new ANTLRInputStream(code);		
		CLexer lexer = new CLexer(charStream);
		CommonTokenStream tokens = new CommonTokenStream(lexer);
		TokenStreamRewriter rewriter = new TokenStreamRewriter(tokens);
		CParser parser = new CParser(tokens);
		
		// Parse and get the root of the parse tree
		ParserRuleContext tree = parser.compilationUnit();

		DeclarationTrimmer dCleaner = new DeclarationTrimmer(parser, rewriter);
		
		ParseTreeWalker walker = new ParseTreeWalker();
		walker.walk(dCleaner, tree);
		
		return rewriter.getText();
	}

	private static LinkedList<VariableDecl> extractLocalVariables(String code) {
		System.out.println("\n\nExtracting local variables");
		
		CharStream charStream = new ANTLRInputStream(code);
		CLexer lexer = new CLexer(charStream);
		CommonTokenStream tokens = new CommonTokenStream(lexer);
		CParser parser = new CParser(tokens);
		
		// Parse and get the root of the parse tree
		ParserRuleContext tree = parser.compilationUnit();

		ParseTreeWalker walker = new ParseTreeWalker();
		LocalVariableExtractor extractor = new LocalVariableExtractor(parser);
		walker.walk(extractor, tree);
		
		System.out.println("Discovered local variables:");
		for(VariableDecl var : extractor.ret) {
			System.out.println(var.type + ", " + var.name + ", " + var.value);
		}
		return extractor.ret;
	}

	// Get a String representation of the input code
	private static String getCode(String gotoFilePath) throws FileNotFoundException {
		Scanner c = new Scanner(new File(gotoFilePath));
		String res = "";
		while(c.hasNext()) {
			res += c.nextLine();
			res += "\n";
		}
		c.close();
		return res;
	}
}
