(* ::Package:: *)

(* Mathematica Package *)

BeginPackage["functions`"]
(* Exported symbols added here with SymbolName::usage *)  

verticalTurn::usage="
verticalTurn[rubik_]
turn the Rubik cube clockwise around the vertical axis
";

horizontalTurn::usage="
horizontalTurn[rubik_]
turn the Rubik cube counterclockwise around the horizontal axis (downwards)
";

faceTurn::usage="
faceTurn[rubik_]
turn the face of the Rubik cube counterclockwise
";

Begin["`Private`"] (* Begin Private Context *) 

verticalTurn[rubik_]:=Module[{turnV,facePermutation={6,4,1,7,2,8,5,3},inversefacePermutation,rubikres},
inversefacePermutation=Permute[facePermutation,{8,7,6,5,4,3,2,1}];
turnV=Permute[Table[{1,2,3,4,5,6,7,8}+8i,{i,0,5}],{5,6,3,4,2,1}];
turnV[[3]]=Permute[turnV[[3]],inversefacePermutation];
turnV[[4]]=Permute[turnV[[4]],facePermutation];
turnV=Flatten[turnV];
rubikres=Permute[rubik,turnV]
;rubikres]

horizontalTurn[rubik_]:=Module[{turnH,facePermutation={6,4,1,7,2,8,5,3},rubikres},
turnH=Join[
	Flatten[Permute[Table[facePermutation+8i,{i,0,4}], {4,3,1,2,5}]],Permute[Table[8*5+i,{i,8}],
	facePermutation]];
rubikres=Permute[rubik,turnH]
;rubikres]

faceTurn[rubik_]:=Module[{a=Table[{1,2,3,4,5,6,7,8}+8i,{i,0,5}],b,facePermutation={6,4,1,7,2,8,5,3},inversefacePermutation},
inversefacePermutation=Permute[facePermutation,{8,7,6,5,4,3,2,1}];
a[[1]]=Permute[a[[1]],inversefacePermutation];
b=a;
a[[4]]=ReplacePart[a[[4]], {1 -> b[[6]][[8]], 4 -> b[[6]][[5]],  6 -> b[[6]][[3]]}];
a[[3]]=ReplacePart[a[[3]], {1 -> b[[5]][[1]], 4 -> b[[5]][[4]],  6 -> b[[5]][[6]]}];
a[[6]]=ReplacePart[a[[6]], {3 -> b[[3]][[6]], 5 -> b[[3]][[4]],  8 -> b[[3]][[1]]}];
a[[5]]=ReplacePart[a[[5]], {1 -> b[[4]][[1]], 4 -> b[[4]][[4]],  6 -> b[[4]][[6]]}];
;Permute[rubik, Flatten[a]]]

End[] (* End Private ContextPermute[rubik, Flatten[a]]*)

EndPackage[]
