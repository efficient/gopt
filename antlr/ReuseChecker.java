import java.util.LinkedList;
import java.util.List;

import org.antlr.v4.runtime.TokenStream;

public class ReuseChecker extends CBaseListener {
	CParser parser;
	TokenStream tokens;
	Debug debug;
	List<String> myVars;	// Local vars used by our compiler	
	
	public ReuseChecker(CParser parser) {
		this.parser = parser;
		tokens = parser.getTokenStream();
		this.debug = new Debug();
		
		myVars = new LinkedList<String>();
		myVars.add("I");
		myVars.add("batch_rips");
		myVars.add("iMask");
		myVars.add("temp_index");
	}

	// primaryExpression is a variable, a constant, or an expression in braces.
	// In a->b and a.b, b is not a primary expression, but a is.
	@Override
	public void enterPrimaryExpression(CParser.PrimaryExpressionContext ctx) {
		String primaryExpression = debug.btrText(ctx, tokens);
		if(myVars.contains(primaryExpression)) {
			System.err.println("ERROR: forbidden expression " + primaryExpression + " appears in code");
			System.exit(-1);
		}
	}
}
