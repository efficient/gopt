import org.antlr.v4.runtime.TokenStream;
import org.antlr.v4.runtime.TokenStreamRewriter;

public class IncludeInserter extends CBaseListener {
	CParser parser;
	TokenStream tokens;
	TokenStreamRewriter rewriter;
	Debug debug;
	int numEntries = 0;
	
	public IncludeInserter(CParser parser, TokenStreamRewriter rewriter) {
		this.parser = parser;
		tokens = parser.getTokenStream();
		this.rewriter = rewriter;
		this.debug = new Debug();
		this.numEntries = 0;
	}
	
	// As TokenStreamRewriter only works inside Listeners, we put the code
	// for inserting the include here. For this to work, either there should be only
	// one function definition in the input code, or the cleanup should be
	// idempotent.
	@Override
	public void enterFunctionDefinition(CParser.FunctionDefinitionContext ctx) {
		if(numEntries != 0) {
			System.err.println("ERROR: IncludeInserter entered twice. Aborting.");
			System.exit(-1);
		}
		numEntries ++;
		
		rewriter.insertBefore(ctx.start, "#include \"fpp.h\"\n");
	}
}
