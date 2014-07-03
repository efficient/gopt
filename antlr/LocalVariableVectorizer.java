import java.util.LinkedList;

import org.antlr.v4.runtime.TokenStream;
import org.antlr.v4.runtime.TokenStreamRewriter;

public class LocalVariableVectorizer extends CBaseListener {
	CParser parser;
	TokenStream tokens;
	TokenStreamRewriter rewriter;
	Debug debug;
	LinkedList<VariableDecl> localVariables;
	int numPrinted = 0;
	
	public LocalVariableVectorizer(CParser parser, 
			TokenStreamRewriter rewriter, LinkedList<VariableDecl> localVariables) {
		this.parser = parser;
		tokens = parser.getTokenStream();
		this.rewriter = rewriter;
		this.debug = new Debug();
		this.localVariables = localVariables;
		this.numPrinted = 0;
	}
	
	// primaryExpression is a variable, a constant, or an expression in braces.
	// In a->b and a.b, b is not a primary expression, but a is.
	@Override
	public void enterPrimaryExpression(CParser.PrimaryExpressionContext ctx) {
		String primaryExpression = debug.btrText(ctx, tokens);
		// debug.println("Primary Expression: " + primaryExpression);
		VariableDecl vd = new VariableDecl("", primaryExpression, "");
		if(localVariables.contains(vd)) {
			// Pretty printing: print 10 replaced local-vars per line
			if(numPrinted % 10 == 9) {
				debug.print(primaryExpression + "\n");
			} else {
				debug.print(primaryExpression + ", ");
			}
			
			numPrinted ++;
			rewriter.replace(ctx.start, primaryExpression + "[I]");
		}
	}
	
}
