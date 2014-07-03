import java.util.List;

import org.antlr.v4.runtime.TokenStream;
import org.antlr.v4.runtime.TokenStreamRewriter;
import org.antlr.v4.runtime.tree.ParseTree;

public class ForeachDeleter extends CBaseListener {
	CParser parser;
	TokenStream tokens;
	TokenStreamRewriter rewriter;
	Debug debug;
	boolean successful = false;
	
	public ForeachDeleter(CParser parser, TokenStreamRewriter rewriter) {
		this.parser = parser;
		tokens = parser.getTokenStream();
		this.rewriter = rewriter;
		this.debug = new Debug();
	}

	@Override
	public void enterIterationStatement(CParser.IterationStatementContext ctx) {
		CParser.CompoundStatementContext csc = new CParser.CompoundStatementContext(null, 0);
		
		if(ctx.getText().startsWith("foreach")) {
			debug.println("Found foreach. Deleting.");
			
			int foreachStart = ctx.start.getTokenIndex();
			int loopStart = -1, loopEnd = -1;	// Token indices of foreach's { and }
			
			List<ParseTree> subtree = ctx.children;
			for(ParseTree child: subtree) {
				if(child.getClass().equals(csc.getClass())) {
					loopStart = child.getSourceInterval().a;
					loopEnd = child.getSourceInterval().b;
				}
			}
			
			for(int i = foreachStart; i <= loopStart; i ++) {
				rewriter.delete(i);
			}
			rewriter.delete(loopEnd);
			successful = true;
		}
	}	
}
