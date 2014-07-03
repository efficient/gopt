import org.antlr.v4.runtime.TokenStream;
import org.antlr.v4.runtime.TokenStreamRewriter;

public class CodeCleaner extends CBaseListener {
	CParser parser;
	TokenStream tokens;
	TokenStreamRewriter rewriter;
	Debug debug;
	int numEntries = 0;
	
	public CodeCleaner(CParser parser, TokenStreamRewriter rewriter) {
		this.parser = parser;
		tokens = parser.getTokenStream();
		this.rewriter = rewriter;
		this.debug = new Debug();
	}

	// As TokenStreamRewriter only works inside Listeners, we put the code
	// for code cleanup here. For this to work, either there should be only
	// one function definition in the input code, or the cleanup should be
	// idempotent.
	@Override
	public void enterFunctionDefinition(CParser.FunctionDefinitionContext ctx) {
		if(numEntries != 0) {
			System.err.println("ERROR: CodeCleaner entered twice. Aborting.");
			System.exit(-1);
		}
		numEntries ++;
		
		debug.println("Running cleanup from function definition: `" + 
				debug.btrText(ctx.declarator(), tokens));
		int numTokens = tokens.size();
		
		// Remove orphaned semicolons
		for(int i = numTokens - 1; i >= 0; i --) {
			if(tokens.get(i).getText().contentEquals(";")) {
				boolean remove = true;
				
				for(int j = i - 1; j >= 0; j --) {
					String t = tokens.get(j).getText();
				
					if(t.contains(" ") || t.contains("\t")) {
						continue;
					} else if(t.contentEquals("\n")) {
						break;
					} else {
						remove = false;
						break;
					}
				}
				
				if(remove) {
					rewriter.delete(i);
				}
			}
		}
	}
}
