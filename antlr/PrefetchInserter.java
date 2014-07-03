import java.util.List;

import org.antlr.v4.runtime.TokenStream;
import org.antlr.v4.runtime.TokenStreamRewriter;
import org.antlr.v4.runtime.tree.ParseTree;


public class PrefetchInserter extends CBaseListener {
	CParser parser;
	TokenStream tokens;
	TokenStreamRewriter rewriter;
	Debug debug;
	int nextLabel = 1;
	
	public PrefetchInserter(CParser parser, TokenStreamRewriter rewriter) {
		this.parser = parser;
		tokens = parser.getTokenStream();
		this.rewriter = rewriter;
		this.debug = new Debug();
	}

	@Override
	public void enterPostfixExpression(CParser.PostfixExpressionContext ctx) {
		if(ctx.getText().startsWith("PREFETCH")) {
			int start = ctx.start.getTokenIndex();
			int stop = ctx.stop.getTokenIndex();
			if(stop == start) {		// Only "PREFETCH"
				return;
			}
			
			if(ctx.getChildCount() != 4) {
				System.err.println("ERROR: Wrong use of PREFETCH(1). Aborting");
				System.exit(-1);
			}
			
			String ch1 = ctx.getChild(1).getText();
			String ch3 = ctx.getChild(3).getText();
			
			if(!(ch1.contentEquals("(") && ch3.contentEquals(")"))) {
				System.err.println("ERROR: Wrong use of PREFETCH (2). Aborting");
				System.exit(-1);
			}
			
			debug.println("Found PREFETCH. Inserting PSS and label.");
			rewriter.replace(start, "PSS");
			rewriter.insertBefore(stop, ", 0, 0");
			
			// Find the ";" after the PREFETCH statement. Valid AST ensures that
			// there is one
			int semicolonIndex = stop;
			for(; semicolonIndex < tokens.size(); semicolonIndex ++) {
				if(tokens.get(semicolonIndex).getText().contentEquals(";")) {
					break;
				}
			}
			
			rewriter.insertAfter(semicolonIndex, "\nlabel_" + nextLabel + ":\n");
			nextLabel ++;
		}
	}
}
