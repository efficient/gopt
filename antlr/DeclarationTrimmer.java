import org.antlr.v4.runtime.TokenStream;
import org.antlr.v4.runtime.TokenStreamRewriter;

public class DeclarationTrimmer extends CBaseListener {
	CParser parser;
	TokenStream tokens;
	TokenStreamRewriter rewriter;
	Debug debug;
	
	public DeclarationTrimmer(CParser parser, TokenStreamRewriter rewriter) {
		this.parser = parser;
		tokens = parser.getTokenStream();
		this.rewriter = rewriter;
		this.debug = new Debug();
	}

	
	// Deletes all declarationSpecifiers. Deletes initDeclarators that
	// don't have an initializer.
	// **
	// declaration ~ declarationSpecifiers initDeclaratorList? ';'
	// initDeclarator ~ declarator | declarator '=' initializer
	@Override
	public void enterDeclaration(CParser.DeclarationContext ctx) {
		
		String declarationSpecifier = debug.btrText(ctx.declarationSpecifiers(), tokens);
		rewriter.delete(ctx.declarationSpecifiers().start, ctx.declarationSpecifiers().stop);
		debug.println("DeclarationTrimmer deleting declarationSpecifier: `" + 
			declarationSpecifier + "`" );
		
		// Delete the mandatory space after the declarationSpecifier
		int stopIndex = ctx.declarationSpecifiers().stop.getTokenIndex();
		debug.println("DeclarationTrimmer deleting useless " + 
				tokens.get(stopIndex + 1).getText() + " after " + declarationSpecifier);
		rewriter.delete(stopIndex + 1);
		
		deletePointerPrefix(ctx.initDeclaratorList());
		deleteNonInitializedDeclarators(ctx.initDeclaratorList());
	}
	
	// Delete the * prefix from pointer initialization.
	// For example, int *a = A; is transformed to a = A;
	private void deletePointerPrefix(CParser.InitDeclaratorListContext ctx) {
		CParser.DeclaratorContext dc = ctx.initDeclarator().declarator();
		String name = dc.getText();
		if(name.contains("*")) {
			String trimmedName = name.replaceAll("[*]+", "");
			debug.println("\tDeclarationTrimmer deleting *s from " + name);
			
			int lo = dc.getStart().getTokenIndex();
			int hi = dc.getStop().getTokenIndex();
			for(int i = lo; i <= hi; i ++) {
				if(tokens.get(i).getText().contains("*")) {
					rewriter.delete(i);
				} else {
					rewriter.replace(i, trimmedName);
				}
			}
		}
	}


	// Delete all initDeclarators from the initDeclaratorsList that have an
	// empty initializer
	private void deleteNonInitializedDeclarators(CParser.InitDeclaratorListContext ctx) {
		CParser.InitDeclaratorContext idc = ctx.initDeclarator();
		CParser.InitializerContext ic = idc.initializer();
		if(ic == null) {
			debug.println("\tLocalVariableReplacer deleting non-initialized declarator: `" + 
					debug.btrText(idc, tokens) + "`");
			rewriter.delete(idc.start, idc.stop);
			
			// Delete commas and spaces after the non-initialized declarator
			int stopIndex = idc.stop.getTokenIndex();
			for(int i = stopIndex + 1; i < tokens.size(); i ++) {
				String tokenString = tokens.get(i).getText();
				if(tokenString.contains(" ") || tokenString.contains(",")) {
					debug.println("\tLocalVariableReplacer deleting useless `" + tokenString + "`");
					rewriter.delete(i);
				} else {
					break;
				}
			}
		}
		
		if(ctx.initDeclaratorList() == null) {
			return;
		}
		
		deleteNonInitializedDeclarators(ctx.initDeclaratorList());
	}
}
