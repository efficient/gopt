import java.util.LinkedList;

import org.antlr.v4.runtime.TokenStream;
import org.antlr.v4.runtime.TokenStreamRewriter;

public class DeclarationInserter extends CBaseListener {
	CParser parser;
	TokenStream tokens;
	TokenStreamRewriter rewriter;
	Debug debug;
	LinkedList<VariableDecl> localVariables;
	int numEntries = 0;
	
	public DeclarationInserter(CParser parser, 
			TokenStreamRewriter rewriter, LinkedList<VariableDecl> localVariables) {
		this.parser = parser;
		tokens = parser.getTokenStream();
		this.rewriter = rewriter;
		this.debug = new Debug();
		this.localVariables = localVariables;
		this.numEntries = 0;
	}
	
	// As TokenStreamRewriter only works inside Listeners, we put the code
	// for inserting lv declarations here. For this to work, either there should be only
	// one function definition in the input code, or the cleanup should be
	// idempotent.
	@Override
	public void enterFunctionDefinition(CParser.FunctionDefinitionContext ctx) {
		if(numEntries != 0) {
			System.err.println("ERROR: DeclarationInserter entered twice. Aborting.");
			System.exit(-1);
		}
		numEntries ++;
		
		// State maintainance code at the beginning 
		String initCode = 	"\tint I = 0;          // batch index\n" + 
							"\tvoid *batch_rips[BATCH_SIZE];       // goto targets\n" +
							"\tint iMask = 0;      // No packet is done yet";

		// Declare all local variables
		String lvDeclarations = "";
		for(VariableDecl vdecl : localVariables) {
			lvDeclarations = lvDeclarations + "\t" + vdecl.arrayDecl() + "\n";
		}
		
		// State maintainance code at the end
		String endCode = 	"end:\n" +
							"\tbatch_rips[I] = &&end;\n" +
							"\tiMask = SET(iMask, I);\n" + 
							"\tif(iMask == (1 << BATCH_SIZE) - 1) {\n"+
							"\t\treturn;\n" +
	    					"\t}\n"+   
	    					"\tI = (I + 1) & BATCH_SIZE_;\n"+
	    					"\tgoto *batch_rips[I];\n";
		
		debug.println("Inserting local var declarations from function definition: `" + 
				debug.btrText(ctx.declarator(), tokens));
		CParser.CompoundStatementContext csx = ctx.compoundStatement();
		
		// The first token of the compoundStatement is '{'
		rewriter.insertAfter(csx.start, "\n" + initCode + "\n\n" + lvDeclarations);
		
		// The last statement of the compoundStatement is '}'
		rewriter.insertBefore(csx.stop, "\n" + endCode + "\n");
	}
	
}
