import java.util.LinkedList;

import org.antlr.v4.runtime.TokenStream;

public class LocalVariableExtractor extends CBaseListener {
	CParser parser;
	TokenStream tokens;
	LinkedList<VariableDecl> ret;		// Type, Identifier
	Debug debug;
	
	public LocalVariableExtractor(CParser parser) {
		this.parser = parser;
		tokens = parser.getTokenStream();
		ret = new LinkedList<VariableDecl>();
		this.debug = new Debug();
	}

	 // declaration ~ declarationSpecifiers initDeclaratorList? ';'
	 // initDeclarator ~ declarator | declarator '=' initializer
	@Override
	public void enterDeclaration(CParser.DeclarationContext ctx) {
		// The type of the declaration (for example, volatile int*)
		String declarationSpecifier = debug.btrText(ctx.declarationSpecifiers(), tokens);
		debug.println("LocalVariableExtractor found declarationSpecifier: `" + 
				declarationSpecifier + "`" );
		
		// The identifiers declared	with this declarationSpecifier
		extractDeclarators(declarationSpecifier, ctx.initDeclaratorList()); 
	}

	// Extract the declarators in the declaration. The declaration has a list
	// of initDeclarators, which we can pass recursively. It's hard to do this 
	// iteratively because initDeclaratorList does not actually expose a List.
	private void extractDeclarators(String declarationSpecifier,
			CParser.InitDeclaratorListContext initDeclaratorList) {
		if(initDeclaratorList == null) {
			return;
		}
		String name = initDeclaratorList.initDeclarator().declarator().getText();
		String value = "";
		if(initDeclaratorList.initDeclarator().initializer() != null) {
			value = initDeclaratorList.initDeclarator().initializer().getText();
		}
		
		VariableDecl var = new VariableDecl(declarationSpecifier, name, value);
		ret.addLast(var);
		debug.println("\tLocalVariableExtractor found declaration: " + var.toString());

		extractDeclarators(declarationSpecifier, initDeclaratorList.initDeclaratorList());
	}
}