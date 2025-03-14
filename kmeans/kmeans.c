/*                                                                                                |
KMEANS.C- An approach to using the K-Means Algorithm to cluster hashed vectors around a number of |
centroids, or central points. The file imports functions previously implemented by LightSys that
performs hashing and vectorization on a string, and eventually the cosine distince (dot product of
a and b divided by the normalization of a * normalization of b) to add it to each centroid. Several
of the methods below were used from LightSys's Centrallix platform in order to have consistency
from our code going back to the actual data stored in Kardia.

--NOTE: It is worth noting that the imported fuctions from https://github.com/LightSys/centrallix/blob/utf-8/centrallix/expression/exp_functions.c consistenly used the unsigned short data type. However, due to issues with accuracy in
clustering and previously performed integer division, we have opted to change most of the values
used to type double for increased precision. 
*/

/*
Explanation: The K-means clustering algorithm, in this example, takes a long list of name strings, pulled from the p_partner table, and assigns a random number within a small scope near the origin to be centroids for the clusters. Then, after creating an array of doubles called vectors, it checks the distance between each centroid, and simply checks for and returns the centroid with the closest distance. Finally, after all points have been assigned, each centroid point is updated as the average of all points in the cluster. More details can be found at https://www.geeksforgeeks.org/k-means-clustering-introduction/
*/

#include <ctype.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Things to fix:
// Optimize Data Conversions
// Add documentation
// Free memory

// Clustering Parameters
#define SEED time(NULL) // Set the seed for random numbers- a null timestamp
#define K 64           // Number of clusters
#define MAX_ITER 16  // Maximum iterations

// Test Data
char* dataset[] = {"Torres","Winslow","Saunders","Moody","Salsmhb","Pendleton","Mathews","Aytou","Xxueinhyly","Giordano","Owen","Dilley","Esaflaiaf","Nynananrn","Ktaggvie","Smith","Hdboop","Nelson","Cartwright","Mike","Mares","Engelhardt","Rteisiw","Nythra","Burnett","Brown","Zodifr","Rose","Hare","Navarra","Depaz","Brooks","Toler","Estelle","Kelleher","Mercado","Carver","Acnbdh","Rojo","Austin","Smith","Osofobyn","Hbotffrfyah","Diaz","Sheehan","Aquni","Young","Stanfield","Ahotuahhtl","Garcia","Fernandez","Valdez","Smith","Brown","Tywpbmr","Lake","Uelcychwo","Randolph","Ramirez","Nona","Templeman","Rebaasinn","Sirirodro","Ononnn","Fowler","Murphy","Rhodes","Arnold","Balderas","Greer","Graham","Ward","Tucker","Welker","Ueiicxhsbtfx","Brown","Todd","Bonnette","Aycock","PfneifbNio","Woodrow","Croviclds","Hon","Roberts","Humphrey","Wrupawehh","Mspzsitaurl","Ingle","Inegaiasujo","Ahaqonau","Thomas","Squires","Anekopraf","Uqiot","Hu","Wilson","Duffy","Beaulieu","Gfghmsokdf","Macey","King","Gomer","Lindsey","Graham","Chrache","Edelman","Alexander","Morales","Benard","Coyuoy","Rich","Tut","Ritter","Naiamuvvat","Fink","Ejoofienajers","Cleskomeu","Mccarty","Caudill","Little","Johnson","Shearin","Miller","Lopez","Stansfield","Aubrey","Hill","Rosenblatt","Eldridge","Giordano","Ewen","Watts","Smith","Headley","Shepard","Llanos","Travis","Craig","Rebaasinn","Fields","Good","Engler","Barker","Ioalsoqr","Ostityh","Hackbarth","Hughes","Torres","Proctor","Porter","Berger","Aaron","Nsennsyf","Stump","Lucier","Freeman","Hancock","Guavooegwaool","Kelleher","Holeman","Harris","Welker","Yoder","Rose","Gilson","Thlht","Xused","Ydulnrtnoe","Ngosqu","Phglptfwhmels","Yongihutej","Jean","Omlnmgwi","Mclean","Pringle","Evans","Siasomm","Ovighsmert","Orozco","Ramsey","Eastman","Cowling","Ishi","Eldridge","Wright","Edmiston","Duke","Erickson","Otwohis","Ollpwsno","Llanos","Tiwei","Watts","Hbotffrfyah","Oghot","Craig","Ijdadmoq","Hlcrefce","Thompson","Marshall","Nnoho","Feder","Theriault","Hounocae","Cowling","Collard","Cbwsvoosy","Kelly","Jennings","Mcdaniel","Leblanc","Nioidcwc","Vefospr","Uhiat","Castleberry","Kirklin","Valdez","Mattison","Diabled","Mcelwee","Huni","Oho","Oxisnaf","C'Gsmnh","Durr","Mark","Durand","Cummings","Limon","Johnson","Fields","Ingram","Prnaoldort","Meyer","Robinson","Kut","Smith","Anna","Graham","Paquin","Abernathy","Dickey","Mullaney","Newman","Berger","Roundtree","Bradley","Mwkcit","Hayden","Ufranua","Erickson","Shnplgruee","Mcvey","Eldridge","Wages","Smith","Tusp","Hicks","Fwaswisu","Winningham","Wayswywn","Fsgfekj","Wills","Black","Strickland","Edwards","Sardina","Erickson","Link","Nichols","Wingard","Berger","Yapuh","Ipua","Auqhsac","Ubeta","Palermo","Fwaswisu","Welker","Cichejiec","Llanos","Raiwwawlt","Pnhlct","Todaro","Nioidcwc","Francis","Rivera","Mcguire","Weiner","Galiano","Zzerninto","Kipii","Hxddld","Fwaswisu","Devveoswe","Cowling","Sullivan","Hfpkaeaoat","Jepsen","Auten","Xah","Young","Oeweh","Hatfield","Nynananrn","Pearson","Pipshc","Zoccwtsiodot","Benavides","Freeman","Cutler","Brown","Se","Ddojtxgmn","George","Casteel","Padro","Servantes","Giblin","Sowghtoane","Engler","Welker","Martin","Janis","Zyhlbe","Martinez","Wiese","Tiehesquua","Mcnaughton","Bower","Juhl","Weiner","Hall","Jyvenu","Owens","Ngifx","Smekate","Hill","Wisdom","Wiezite","Watts","Engler","Marshall","Gilbert","Plasoh","Hill","Sheldon","Jaramillo","Tanoexua","Clark","Reed","Hayden","Iwtyac","Tctadrnto","Aiaswetie","Ayala","Echaiaqoo","Cheryl","Brown","Munoz","Nelson","Wood","Graham","Evans","Colley","Scriber","Valdez","Lee","Uashuax","TdnsobPoka","Moody","Sousa","Opoluoy","Thomas","Cheryl","Bryan","Zito","Ayala","Flint","Xrsfldgbs","Green","T'Eits","Wu","Yurtgkjs","Gilson","Jones","Oxuxe","Ofiemsdd","Gautreaux","Ohvte","Perez","Slisajo","Ryan","Gotagi","Morales","Sutyramm","Nfrzwesdl","Mcintyre","Mcvey","Engler","Longo","Hammond","Erickson","Russell","Black","Wyche","Philip","Hammond","Headley","Weller","Jackson","Miller","Bottoms","King","Boyd","Bashaw","Aiello","Chaney","Davis","Chaney","Ouchettisei","Smith","Bryan","Velarde","Xminn","Kerr","Levesque","Vonlozna","Hshni","Noble","Curtis","Ruta","Ightebr","Bridget","Gomer","Carr","Mcneill","Meyer","Alley","Hill","Deborah","Xniplt","Lopez","Navarra","Reamer","Rene","Espinoza","Llclopn","Harris","Chance","Bevilacqua","Clark","Hilliard","Klmlloah","Anthony","Emert","Meaux","Thomas","Suagdmoul","Sheehan","Furr","Uhiat","Grote","Croft","Summers","Ingram","White","Burk","Ryan","Williams","Raleigh","Dunlap","Mjnotxtukh","Mejia","Hackbarth","Donahue","Lindsey","Knoqlihzrs","Priegloosa","Garcia","Faulkner","Reams","Baird","Bradley","Durr","Lee","Hmaamgcn","Moore","Kiyonjtfk","Welker","Motes","Stewart","Ryan","One","Dilley","Arthur","Nocddwhh","Miller","Patea","Seoaical","Dunning","Faulkner","Carver","Vafnnhr","Ortwlfkniy","Ceglelrzsx","Nkhlh","Bohxiwsy","Berger","Abrams","Pintajoi","Istuolg","Hurlbut","Fluharty","Ndwiibaiui","Bryant","Mcvey","Green","Bancdzr","Bullock","Yates","Conethr","Williams","Ardoin","Stanton","Thomas","Jean","Allshouse","Moody","Aiavu","Michelle","Inato","Davis","John","Berger","Carr","Ditasni","Gilson","Gomer","Naghti","Hese","Berger","Rousseau","Cloer","Martinez","Hbotffrfyah","Jenson","Smith","Bunker","Russell","Collins","Schmidt","Iy","Unat","Valdez","Ky","Travis","Arfajyhb","Tunhjtv","Ypeccintis","Oafyr","Vtpgmlh","Ydulnrtnoe","Ssss","Nynananrn","Hundley","Lucier","Linda-Brown","Dart","Vergara","Costello","Ford","Ibow","Marilyn","Lee","Bradshaw","Hsdt","Leftwich","Gurley","Patrick","Williams","Pritts","Berger","Adams","Kevin","Wise","Berger","Rose","Tati","Hansen","Brinn","Brown","Martin","Whijas","Miller","Weiner","Cowan","Icbq","White","Moses","Allan","Ross","Hayden","Eslick","Rademacher","Hyman-Spencer","Ebhprwagt","Gene","Rousseau","Tozier","Mongillo","Berger","Utighmat","Mai","Brock","Cancel","Coy","Williams","Oziep","Iviks","Miller","Flanagan","Eeduh","Faust","Hunter","Neeriy","Osacac","Nielsen","Nucetax","Castillo","Behrens","Bridges","Unat","Edwards","Thomas","Brecht","Nelson","Engler","Williams","Levesque","JpBsohelh","tx Gonzalez","Lawson","Smith","Allard","C'Xtuw","Bevilacqua","Lawson","Obrien","Babcock","Wofford","Faulkner","Rogers","Ingram","Garza","Llgsji","Mclaughlin","Allen","Levitt","Lopez","Pagey","Hayden","Ooiabeil","Gautreaux","Omyns","Velarde","Blank","Brown","Schilling","Byrd","Gomer","Leeds","Brooks","Mcginnis","Sofeatretau","Smith","Dykes","Weiner","Wmrdtrzh","Znihadeocwy","Graham","Ronald","Tobey","Nuostoqpe","Gibson","Ridley","Kelleher","Bmdajags","Yimzysva","Scriber","Daniels","Saunders","Allen","Tpnifzomm","Pefpahrf","Costa","Timothy Like Frln","Coleman","Brock","Gould","Harris","Williams","Mortensen","Bush","Pitts","Calvo","Cook","Xwcfqnrsiy","Isque","Taylor","Berger","Nscva","Turner","Mapp","Mcdorman","Berger","Fitzpatrick","Deosob","Todaro","Carranza","Aledyjiisdci","Shaw","Enxbya","Fabhtgef","Harmon","Waring","Bednar","Baker","Hughes","Frank","William","Bush","French","Hughes","Mathews","Durand","Roof","Trfydqwant","Brafford","Mwprymndhl","Yorke","Moulton","Hopkins","Vvurqbagbiy","Ddighrsif","Felix","Michael","Castle","Bsricte","Engler","Graham","Surrett","Ehrhardt","Oeweh","Thomas","Reed","Atrainnr","Tannehill","Sddgif","Valdez","Tsnbijh","Sosa","Rigsby","Oafyr","Faulkner","Ecgtnt","Bancdzr","Nitaji","Faulkner","Tello","Badger","Harris","Hansen","Scriber","Thompson","Lanning","Grant","Nynananrn","Dugan","Nneaen","Chaney","Oafyr","Hyatt","Eley","Brooks","Chiesa","French","Harris","Lopez","Tfsd","Lindsey","Ixis","Mai","Casteel","Kevin But","S'Lfkruw","Buzard","Randolph","Alexander","Harris","Theresa","Moore","Gomer","Pitcher","Nrpucqnsf","Wright","Painter","Ellison","Fxrryv","Oqu","Coutu","Osans","Meyer","Carpenter","Warren","Dorsey","Gautreaux","Bridges","Aaron","Turner","Oneiqnmui","Wyche","Seymour","Leftwich","Ozde","Jaghtika","Kidwell","Rodas","Mopw","Welker","Rose","Tylor","Aparicio","Tisiti","Crain","Berger","Roller","John","Blank","Fuller","Frisby","Yoshida","Usuza","Engler","Hayden","Nwuue","Maguire","Najef","Newcomb","Rsmfkya","Edwards","Lin","Larson","Gonzalez","Figueiredo","Roundtree","Smith","Bendhatir","Guillen","Edwards","Squires","Evans","Qini","Jason","Roundtree","Limon","Ho","Zito","Esudi","Roberts","Oliverio","Duncan","A Duran","Mcdade","Ingraham","Ross","Hyder","Saunders","Usa","Xmtb","Barns","Tozier","Younce","Alonzo","Wyche","Cullen","Ggi","Chaney","Mccullough","Kramer","Lombard","Bullock","Kershner","John","Jones","Black","Bradley","Poole","Barns","Blount","Bjczst","Jesaw","Avoysaek","Uslahey","Xxa","Ward","Pollack","Stovo","Mitchell","Pence","Itdwuydasg","Perkins","Nutapa","Boss","Durand","Lindell","Reed","Uwdffne","Prosser","Stewart","Rasberry","Carroll","Porter","Ecorr","Watson","Nichols","Weiner","Clark","Proctor","Ighqiobbua","Thibowsat","Buchanan","Arner","Jacobs","Russell","Green","Saunders","Mcnaughton","Lindell","Matthew","Otrayca","Beard","Rousseau","Aon","Matheney","Nasun","Qispinui","Rousseau","Lucier","Ioto","Lcotg","Pjrdsalmos","Viofuvv","Orot","Wnhgs","Parker","Ehedmo","Skyyae","Carlson","Kuhl","Valdez","Hyatt","Little","Foor","Hobo","Rjs","Oneiqnmui","Rseyuyisufa","Grace","Lopez","Uscepryny","Smith","Wrusprosp","Seabrook","Odonnell","Xizuezshda","Pugh","Fiejaskai","Nekum","Weiner","Edwards","Ceglelrzsx","Gilbert","Abectes","Aisfotn","Muniz","Barker","Garcia","Allan","Durand","Owjenuqaud","Smith","Styln","Jennings","Ascreoha","Sandra","Estrada","Bevilacqua","I.","Faulkner","Usa","Boyle","Goddard","Ronald","Sullivan","James","Mcnaughton","Avoysaek","Underwood","Monroe","Durr","Quinn","Olmstead","Rytrnd","Alexander","Harden","Lindsey","Paden","Edieflo","Parris","Lee","Wmitt","Hamdan","Harris","Doopfftnh","Cxcvderww","Painter","Dean Dennis","Ilkpa","Saunders","Aiello","Bryan","Rsuminebloi","Hatcher","Feder","Ciosangleta","Wivpibc","Templeman","Chaney","Justin Rrital","Rose","Aydzi","Drake","Jarrett","Owens","Ridley","Bevilacqua","Buckley","Hhbejlg","Salazar","Bull","Montgomery","Oswalt","Jones","Singleton","Riyepaqs","Beendo","Raleigh","Floyd","Woodrow","Moore","Wnta","Jarrett","Ridley","John","Swoe-Manis","Shepard","Garrett","Jones","Custis","Hull","Jacobs Findley","Wright","Padro","Welker","Sehhess","Vaiadonaisso","Romero","Kraft","Nynananrn","Welker","Levesque","Wilson","Holloway","Edwards","Rendon","Graham","Cheryl","Ford","Tuqo","Mashburn","Faulkner","Atrainnr","Cetnb","Brown","Engler","Nelson","Sanford","Reger","Ridley","Jones","Kkasea","Tfsd","Limon","Tfsd","Tdplew","Franklin","Phillips","Miller","Hayes","Niho","Carroll","Grant","Ihmd","Nteosi","Ane","Layjexo","Monk","Barker","Alexander","Findley","Feder","Eldridge","Rodriquez","Flanagan","Sootan","Welker","Harris","Bradley","Ablaele","Tello","Hibt","Phdwnsermi","Wpursjf","Wu","Burk","Schumacher","Flores","Vrsftpd","Eggers","Dilley","Archer","Charles","Erauce","Cheryl","Braaten","Williams","Taylor","Gilson","Edwards","Smiley","Fitzpatrick","Yu","Macey","Parker","Mcintyre","Waterbury","Jackson","Llanos","Ghap","Rogers","Rollin","Tannehill","Dotson","Watts","Foley","Iseddut","Jefferson","Thomas","Fuller","Bradley","Waldron","Rungu","Adkins","Padro","Ttqirqdie","Kiss","Mitchell","oyo og Aaiyth","Osoo","Kuykendall","Ford","Hill","Jepsen","Oevud","Buchanan","Floyd","Eanwct","Otbnbdlgrne","Berger","Ohenmn","Paiasca","Ioh","Fleming","Neumann","Aytou","Avila","Green","Ransom","Dugan","Chaney","Koeller","Ingram","Parris","Moehpds","Boomer","Vela","Hackbarth","Castorena","Rasberry","Larochelle","Yariswo","Wu","Gambrel","Baugh","Singer","Bciedjcht","Nfypobonuy","Sean","Mazza","Meadows","Kennedy","Uosowaf","Yu","Eric","Caudill","Daniel","Scott","Gustafson","Owens","Lewis","Wynn","Matheney","Key","Oliverio","Davis","Thomas","Figueroa","Zibli","Fleming","Kenny","Barker","Floyd","Surrett","Mire","Fletcher","Jauregui","Faulkner","Walter","Rousseau","Beaulieu","Hiwhno","Piscitelli","Garza","Perez","Chance","John","Craig","Leist","Gautreaux","Ca Srxdsi","Patrick","Whubej","Cancel","Floyd","Harrington","Ixikuk","Pellot","Berger","Hackbarth","Jdzrpwe","Orosco","Chance","Peck","Popp","Mecune","Cgyzsongaltcftbn","Dugan","Aytou","Wingard","Bolnsy","Friend","Marvin","N'Hevrdv","Bridges","Elenae","Egaiay","Seuzoche","Ness","Butts","Sumter","Nason","Angli","Nesmic","Daniel","Aidrobpicinrgsv","Toler","Jensen","Salinas","Woods","Brown","Mapp","Stewart","Welker","Harrington","Mgquszn","Tannehill","Watts","Gautreaux","Newcomb","Nciboy","Welker","Akridge","Barker","Eobnsap","Lally","Beagle","Moua","Jepsen","Doopfftnh","Tfhgszv","Oxuxe","Esquivel","Wallis","Ooiabeil","Lozi","Smith","Stvmnidmcnfl","Engler","Lindsey","Aytou","Ahtlibrnw","Park","Hartman","Hkacag","Tiun","Berger","Rovesnetoh","Floyd","Reyes","Ewhu","Hjaw","Otoxisis","Cargill","Woodward","NnzUuoamaf","Hawley","Mclaughlin","Hgaosvzisdk","Winningham","Asasloni Joiner","Williams","Smith","Cramer","Janssen","Agigge","Lui","Chappell","Ters","Bush","Kelleher","Mullis","Haney","Ocmobe","Ekikas-Martin","Eku","Tati","Tuqo","Llanos","Wmlodmft","Jones","Wise","Hunt","Ireofcoe","Giraldo","Cadpaaghd","Hodge","Johnson","David Ugrez","Dayton","Tannehill","Mcvey","Davis","Otaskisa","Ky","Edmiston","Knutsen","Subeyfluisou","Caudill","Shine","Harrington","Hopkins","Richardson","Lewis","Cheryl","Mcmahon","Yfu","Sehhess","Nglyveyuspeb","Bills","Ped","Bryan","Qivas","Opoluoy","Nipiouiyn","Byers","Dopdlavforo","Diaz","Hackbarth","Edhorksa","George","Oaeemrla","Mathis","Gsinb","Horner","Arnold","Bevilacqua","Parker","Laka","Fpfolie","Erickson","Hayden","Ricardo","Harris","Cheryl","Vela","Dilley","Engler","Oceraix","Smith","Gautreaux","Field","Meprabi","Negron","Emmert","T'Eits","Alaqiac","Sddgif","Slayton","Grant","Glen","Mitchell","Barns","Mdsain","Weiner","Smith","Mitchell","Leftwich","Eispugr","Bevilacqua","Bryan","Moody","Oegde","MEDEIROS","Ohowf","Wnnsioloa","Sn. Mcleod","Byrd","Mathis","Jepsen","Philip","Hu","Piscitelli","Carroll","Taylor","Kgsiiqm","Salazar","Wttobnidelqj","Joyner","Acuy","Davis","Httddygiocs","Barker","Baker","Egoo","Blank","Campbell","Jackson","Penney","Enxbya","Rose","Engler","Soaneyhea","Bevilacqua","Ujaiasobi","Cancel","Moua","Tywpbmr","Byers","Leeds","Goss","Moua","Goddard","Stevens","Bryan","Odonnell","Powell ","Cutler","Hunter","Tefrehoo","Harris","Cecil Sakka","Coleman","Whitham","Christine","Yu","Forester","Tharpe","Square","Babcock","Hiwhno","Fruatoat","Mohr","Norman","bsej","Crazmniko","Tovar","Tdiynjy","Williams","Maggio","Halbert","Bevilacqua","Cgaeq","Woods","Mdls","James","Nenwcp","Moore","Okelley","Russell","Faust","Lynch","Dority","Ojikyxu","Wrarheroe","Morrow","FDPR","Feder","Clayton","Coleman","Humphrey","Faulkner","Farrington","Tucker","Golla","Santiago","Smith","Jordan","Baril","Soinuda","Adams","Austin","Weaver","Sadrtie","Ahesabe","Ti","Quinn","Welker","Hughes","Parnell","Tiboajas","Dilley","Smith","Soytosigh","Taylor","Rau","Usy","Ffyavn","Azetoeq","Casteel","Zwghram","Akbar","Mfsfa","Gautreaux","Bush","Robert","Nason","Knoll","Mwtodali","Jimenez","Oeweh","Depaz","SHRIO","Franklin","Ifleetewah","Nelson","Sanchez","Montgomery","Caudill","Dugan","Hansen","Dupak","Aleexau","Carson","Yhashswa","Carroll","Alexander","Russell","Aschoir","S'Lfkruw","Jones","Gautreaux","Rodriguez","Gmoirhqays","Benjamin","Miraglia","Sims","Omhpl","Felix","Igheni","Eectalloyj","Bryan","Hackbarth","Echevarria","Jason","Bevilacqua","Wminlet","Cummings","Johnson","Arthurs","Ypetas","Skinner","Padro","Chuscieqoeru","Eastin","Shields","Bailey","Davis","Zoccwtsiodot","Ayers","Tiwei","Nltiabfga","Smith","Wright","King","Peasun","Mullaney","Gonzales","Little","Leeds","Williams","Lowe","Fitzgerald","Fink","Hayden","Lucier","Anekopraf","Cobb","Hudgins","Oxuxe","Haile","Kiefer","Blank","Justin","Deborah","Bergman","Jenkins","Craig","Marshall","Eastridge","Robinson","Warner","Jason","Soinuda","Itisoff","Beeler","Xorrotiteal","Baldwin","Schrader","Enihp","Cover","Kuhn","Ueduoy","Wyche","Edwards","Gbqsstd","Cain","Tdsszawutbs","Price","Patrick","Atytej","Ztga","Young","Blaney Raap","Sims","Brooks","Hobo","Henry","Acosta","Harper","Harper","Poole","Negio","FDPR","Reif","Corso","Ewyx","Matthews","Ny - UZZOLUYAS","Beck","Harris","Rendon","Pemw","Stephens","Rflkri","Erickson","Gilbert","Friedland","Williams","Hawkins","Michael","Howard","Whubej","Woods","Martin","Avoysaek","Mclaughlin","Odonnell","Oxuxe","Nrsgkban","Lin","Golla","Bevilacqua","Pope","Lavoie","Rieth","Royster","Jackson","Npnzesfti","Hubbard","Gaul","Mcmorris","Mondragon","Farias","Va","Breunig","Sefli","Grrldoradagc","Staten","Casteel","Nuasezisno","West","Steely","Barile","Gbieedm","Tannehill","Lindsey","Dyson","Arin","Graves","Brthhjeda","Davis","Fbojgbocbyt","Jesse","Kelleher","Pack","Hodduy","Seu","Ufrewe","Berger","Nynananrn","Brown","Kelleher","Calvin","Vanish","Powell","Robert","Lopez","Flowers","Squires","Ntvxe","Lhkspfxdtws","Tarlnun","Stcxtss","Glnuhhp","Ward","Durr","Jackson","Bryan","Redman","Scriber","Awuniazu","Bryd","Sharp","Welker","Bunting","Shafer","Rusjoaiit","Ssaso","Nnozi","Ricciardi","Lee","Ocyw","Alonzo","Dean","Summers","Welker","Cook","Ateeciavuo","Eerfhfdztsrp","Albertson","Beaulieu","Rene","Noel","Tozier","Gdnnhocghc","Leftwich","Lindsey","Rhoden","Carroll","Ocasio","Faulkner","BIX","Mcguire","Black","Mcguire","Olarun","Croft","Bevilacqua","Moore","Affefu-Vasquez","Atiwo","Hobson","Ryhvt","Scott","Robinson","Taylor","Keene","Uwixeajiin","Joseph","Daniel","Batabre","Campbell","Dilley","Kruse","Enxbya","Mathis","Horton","Smith","Smiley","Mcdevitt","Gonzales","Davis","Stevens","Zuniga","Martin","Vneayvvan","Tello","Ardoin","Brunson","Mccoy","Moiwidvwma","Hebert","Gautreaux","Cramer","Wilbanks","Adaf","Tillery","Noureus","Biagdqog","Ocasio","Neely","Ne","Sob","Shivers","Tdpibkn","Ghtatuobuor","Mcqueen","Ross","Weiner","Drake","Jackson","Gutierez","Xxa","French","Daut","Greco","Eggers","Atrainnr","Ikeqcs","Costello","Carroll","Thompson","Manning","Singleton","Starks","Ard","Frisby","Sheeran","Baxter","Bancdzr","Bell","Ross","Dallas","Wrobndozb","Keltner","Eep","Eiqoffautee","Primm","Mcvey","Ronald","Valdez","Wssnrc","John","Barajas","Waldrep","Robinson","Rieth","Wolf","Templeman","Allen","Padro","Connelly","Sob","Cummings","Figueroa","Weiner","Scriber","Tucker","N'Hevrdv","Smith","Burk","Edmiston","Moor","Gautreaux","Follansbee","Jennings","Oostuzas","Riosceo","White","Keller","Engler","Nobles","Hbotffrfyah","Richard","Burgher","IhccavaAeshv","Casteel","Smith","Llqsunu","Smith","Greer","Pdaccgu","Griffin","Nlibe","Babcock","Daniel","Reed","Layton","Marshall","Ness","Giordano","Furr","Yprone","Turberville","Feliciano","Murphy","Ross","Ehrhardt","Uagabos","Roman","Benningfield","Cheryl","Mftnlt","Babcock","Cover","Hare","Engler","Perkins","Puacceoske","Eispugr","Kuster","Tellier","Holland","Lewter"};
#define strs dataset

// ====================================
// Centralix Code

// This is the size of the vector table. It is also used in calculating the table indices.
const int EXP_VECTOR_TABLE_SIZE = 251;

/*
 * hash_char_pair
 * This method creates an vector table index based a given character pair. The characters are represented 
 * as their ASCII code points.
 *
 * Parameters:
 * 	num1 : first ASCII code point (double)
 * 	num2 : second ASCII code point (double)
 *
 * Returns:
 * 	vector table index (integer)
 */
int exp_fn_i_hash_char_pair(double num1, double num2) {
    int func_result = round(((num1 * num1 * num1) + (num2 * num2 * num2)) *
                            ((num1 + 1) / (num2 + 1))) -
                      1;
    return func_result % EXP_VECTOR_TABLE_SIZE;
}

/*
 * exp_fn_i_dot_product
 * This method calculautes the dot product of two vectors.
 *
 * Parameters:
 * 	dot_product : the place where the result is stored (double)
 * 	r_freq_table1 : the first vector (double)
 * 	r_freq_table2 : the second vector (double)
 *
 * Returns:
 * 	0	
 */
int exp_fn_i_dot_product(double* dot_product, double* r_freq_table1, double* r_freq_table2) {
    int i;
    for (i = 0; i < EXP_VECTOR_TABLE_SIZE; i++) {
        *dot_product = *dot_product + (r_freq_table1[i] * r_freq_table2[i]);
    }
    return 0;
}

/*
 * exp_fn_i_magnitude
 * This method calculates the magnitude (also known as the normalization) of a vector
 * This is calculated as the square root of all squared elements in the vector
 *
 * Parameters:
 * 	magnitude : the place where the result is stored (double)
 * 	r_freq_table : the vector (double)
 */
int exp_fn_i_magnitude(double* magnitude, double* r_freq_table) {
    int i;
    for (i = 0; i < EXP_VECTOR_TABLE_SIZE; i++) {
        *magnitude = *magnitude + (r_freq_table[i] * r_freq_table[i]);
    }
    *magnitude = sqrt(*magnitude);
    return 0;
}

/*
 * exp_fn_i_frequency_table
 * This method creates a vector frequency table based on a string of characters.
 * This is essentially the hashing algorithm for a string into a vector
 *
 * Parameters:
 * 	table : integer pointer to vector frequency table (double)
 * 	term : the string of characters (char*)
 *
 * Returns:
 * 	0
 */
int exp_fn_i_frequency_table(double* table, char* term) {
    size_t i;
    // Initialize hash table with 0 values
    for (i = 0; i < EXP_VECTOR_TABLE_SIZE; i++) {
        table[i] = 0;
    }

    int j = -1;
    for (i = 0; i < strlen(term) + 1; i++) {
        // If latter character is punctuation or whitespace, skip it
        if (ispunct(term[i]) || isspace(term[i])) {
            continue;
        }

        double temp1 = 0.0;
        double temp2 = 0.0;

        // If previous character is null
        if (j == -1) {
            temp1 = 96;
        }

        // Else character is not null
        else {
            temp1 = (int)tolower(term[j]);
        }

        // If latter character is null
        if (i == strlen(term)) {
            temp2 = 96;
        }

        // Else character is not null
        else {
            temp2 = (int)tolower(term[i]);
        }

        // Else character is not null	// If either character is a number, reassign the code point
        if (temp1 >= 48 && temp1 <= 57) {
            temp1 += 75;
        }

        if (temp2 >= 48 && temp2 <= 57) {
            temp2 += 75;
        }

        // Hash the character pair into an index
        int index = exp_fn_i_hash_char_pair(temp1, temp2);

        // Increment Frequency Table value by number from 0 to 13
        // table[index] += (temp1 + temp2) % 13 + 1;
        table[index] += 1;

        // Move j up to latter character before incrementing i
        j = i;
    }

    return 0;
}

// ====================================
// Vectoring Code

/*
 * check
 * Helper function for compact error handling on library & system function calls.
 *
 * Parameters:
 * 	result : The result of the function we're checking (const int)
 * 	functionName : second ASCII code point (const char*)
 *
 * Returns:
 * 	void
 */
static inline void check(const int result, const char* functionName) {
    if (result != 0) { // An error occured.
        // Create the most descriptive error message we can.
        char errorBuffer[1024];
        sprintf(errorBuffer, "vector.c: Fail - %s", functionName);
        perror(errorBuffer);

        // Exit repeatedly until it works, in case exit gets interupted somehow.
        while (1) exit(result);
    }
}

/*
 * create_vector
 * Allocates memory for vector storage.
 *
 * Parameters:
 *
 * Returns:
 * 	A pointer to the new vector (double*).
 */
static inline double* create_vector() {
    double* vector = malloc(EXP_VECTOR_TABLE_SIZE * sizeof(double));
    if (vector == NULL) {
        perror("Memory allocation failed.\n");
        while (true) exit(-1);
    }
    return vector;
}

/*
 * build_vectors
 * Takes an array of strings (strs) and converts them to vectors.
 * create_vector() is called to allocate memeory for the new vectors.
 *
 * Parameters:
 * 	vectors : The location to store new vectors will be stored (double**).
 * 	strs : The strings used to build the vectors.
 *  num_vectors : The number of vectors to create.
 *
 * Returns:
 * 	0 : success
 */
static inline int build_vectors(double** vectors, char** strs, size_t num_vectors) {
    for (size_t i = 0; i < num_vectors; i++) {
        double* vector = vectors[i] = create_vector();
        check(exp_fn_i_frequency_table(vector, strs[i]), "exp_fn_i_frequency_table");
    }
    return 0;
}

/*
 * similarity
 * Calculates the cosine similarity, aka. the angle between the two vectors
 * in n dimentional space (where n is EXP_VECTOR_TABLE_SIZE). In other words,
 * similarity(A, B) performs the function cos(θ) = (A. B)/(|A| |B|) where A
 * and B are two vectors.
 *
 * Parameters:
 * 	v1 : The first vector being compared (double*).
 * 	v2 : The second vector being compared (double*).
 *
 * Returns:
 *  0 - 1 : where 0 means that the two vectors (and the strings they represent)
 *          have no similarity at all, and 1 means that the two vectors are
 *          identical.
 *
 * Assumptions:
 *  All dimentions of both parameters are assumed to be strictly positive.
 */
static inline double similarity(double* v1, double* v2) {
    // Calculate dot product
    double dot_product = 0;
    check(exp_fn_i_dot_product(&dot_product, v1, v2), "exp_fn_i_dot_product");

    // Calculate magnitudes of each relative frequency vector
    double magnitude1 = 0;
    double magnitude2 = 0;
    check(exp_fn_i_magnitude(&magnitude1, v1), "exp_fn_i_magnitude");
    check(exp_fn_i_magnitude(&magnitude2, v2), "exp_fn_i_magnitude");
    
    // Calculate the similarity score
    return dot_product / (magnitude1 * magnitude2);
}

/*
 * similarity
 * Inverts the cosine similarity such that more similar vectors produce a
 * smaller value instead of a larger value.
 *
 * Parameters:
 * 	v1 : The first vector being compared (double*).
 * 	v2 : The second vector being compared (double*).
 *
 * Returns:
 *  0 - 1 : where 1 means that the two vectors (and the strings they represent)
 *          have no similarity at all, and 0 means that the two vectors are
 *          identical.
 *
 * Assumptions:
 *  All dimentions of both parameters are assumed to be strictly positive.
 */
#define distance(v1, v2) (1.0 - similarity((v1), (v2)))

// Debug helper function to print distances.
#define print_distance(i1, i2)                                    \
    printf(                                                        \
        "Distance from '%s' to '%s': %lf\n",                        \
        dataset[i1], dataset[i2], distance(vectors[i1], vectors[i2]) \
    ); // Debug

/*
 * average_cluster_size
 *
 * Parameters:
 * 	vectors : The vectors in the clusters. (double**)
 *  num_vectors : The number of vectors to create. (int)
 * 	labels : The vectors in the clusters. (int*)
 *
 * Returns:
 * 	0 : success
 */
static inline double average_cluster_size(double** vectors, int num_vectors, int* labels, double** centroids) {
    double cluster_sums[K] = {0};
    double noncluster_sums[K] = {0};
    int cluster_counts[K] = {0};

    // Sum the distance from each vector to its cluster centroid.
    for (int i = 0; i < num_vectors; i++) {
        int label = labels[i];
        double* vector = vectors[i];
        cluster_sums[label] += distance(vector, centroids[label]);
        cluster_counts[label]++;

        for (int j = 0; j < K; j++) {
            if (j == label) continue;
            noncluster_sums[j] += distance(vector, centroids[j]);
        }
    }

    // Calculate the average distance per cluster and then the overall average.
    double overall_avg = 0.0;
    int valid_clusters = 0;
    for (int label = 0; label < K; label++) {
        int cluster_count = cluster_counts[label];
        if (cluster_count > 0) {
            double cluster_size = cluster_sums[label] / cluster_count;
            double noncluster_size = noncluster_sums[label] / (num_vectors - cluster_count);
            overall_avg += cluster_size;
            valid_clusters++;

            printf(
                "> Cluster #%d (x%d): %.4lf (vs. %.4lf).\n",
                label, cluster_count, cluster_size, noncluster_size
            ); // Debug
        }
    }

    // If no clusters have any points, return 0.
    return (valid_clusters > 0) ? (overall_avg / valid_clusters) : 0.0;
}

/*
 * kmeans
 * Executes the k-means clustering algorithm. Selects K random vectors as to
 * be initial centroids. Then, assigns points to the nearest centroid, after
 * which it moves centroids to the center of their assigned points.
 *
 * Parameters:
 *  vectors : The vectors to cluster. (double**)
 *  num_vectors : The number of vectors to cluster. (int)
 *  labels : Stores the final cluster identities of the vectors after
 *           clustering is completed. (int*)
 *  centroids : Stores the locations of the centroids used for the clusters
 *              of the data. (double**)
 *
 * Returns:
 *  void
 *
 * Assumptions:
 *  K is in scope and represents the number of centroids.
 *  MAX_ITER is in scope and represents the max number of clustering iterations.
 *
 * Known Issues:
 *  At larger numbers of clustering iterations, some clusters have a size of
 *  negative infinity. In this implementation, the bug is mitigated by setting
 *  a small number of max iterations, such as 16 instead of 100.
 *  Also, clusters do not apear to improve much after the first iteration, which
 *  puts the efficacy of the algorithm into question, at least to some extent.
 *  However, the resulting clusters do still appear to be useful.
 */
static inline void kmeans(double** vectors, int num_vectors, int* labels, double** centroids) {
    // Select random vectors to use as the initial centroids.
    srand(SEED);
    for (int i = 0; i < K; i++) {
        // Pick a random vector.
        const int random_index = rand() % num_vectors;
        printf("Centroid %d starts at vector %d.\n", i, random_index); // Debug

        // Copy each dimetion from the selected random vector to the current centroid.
        for (int dim = 0; dim < EXP_VECTOR_TABLE_SIZE; dim++) {
            centroids[i][dim] = vectors[random_index][dim];
        }

        print_distance(0, random_index); // Debug
    }
    printf("\n"); // Debug

    // Allocate memory for new centroids
    double** new_centroids = (double**) malloc(K * sizeof(double*));
    for (int i = 0; i < K; i++) {
        new_centroids[i] = create_vector();
    }

    // Main loop
    int* cluster_sizes = (int*) malloc(K * sizeof(int));
    for (int i = 0; i < MAX_ITER; i++) {
        bool changed = false;

        // Reset new centroids
        for (int i = 0; i < K; i++) {
            for (int dim = 0; dim < EXP_VECTOR_TABLE_SIZE; dim++) {
                new_centroids[i][dim] = 0.0;
            }
            cluster_sizes[i] = 0;
        }

        // Assign each point to the nearest centroid
        for (int i = 0; i < num_vectors; i++) {
            double* vector = vectors[i];
            double min_dist = DBL_MAX;
            int best_centroid_label = 0;

            for (int j = 0; j < K; j++) {
                double dist = distance(vector, centroids[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_centroid_label = j;
                }
            }

            if (labels[i] != best_centroid_label) {
                labels[i] = best_centroid_label;
                changed = true;
            }

            // Accumulate values for new centroid calculation
            double* best_centroid = new_centroids[best_centroid_label];
            for (int dim = 0; dim < EXP_VECTOR_TABLE_SIZE; dim++) {
                best_centroid[dim] += vector[dim];
            }
            cluster_sizes[best_centroid_label]++;
        }

        // Update centroids
        for (int j = 0; j < K; j++) {
            if (cluster_sizes[j] > 0) {
                for (int dim = 0; dim < EXP_VECTOR_TABLE_SIZE; dim++) {
                    centroids[j][dim] /= cluster_sizes[j];
                }
            }
        }

        printf("Average cluster size is %.6lf.\n\n", average_cluster_size(vectors, num_vectors, labels, centroids));
        check(fflush(stdout), "fflush");

        // Stop if centroids didn't change
        if (!changed) break;
    }
    
    for (int i = 0; i < num_vectors; i++) {
        double* vector = vectors[i];
        double min_dist = DBL_MAX;
        int best_centroid_label = 0;

        for (int j = 0; j < K; j++) {
            double dist = distance(vector, centroids[j]);
            if (dist < min_dist) {
                min_dist = dist;
                best_centroid_label = j;
            }
        }
        labels[i] = best_centroid_label;
    }
}

int main() {
    // Set stdout to only flush manually with a 64MB buffer.
    setvbuf(stdout, NULL, _IOFBF, 64 * 1000 * 1000);

    int num_vectors = 1975;

    // Allocate ram to store vectors
    double** vectors = malloc(num_vectors * sizeof(double*));
    
    // Build the vectors
    check(build_vectors(vectors, strs, num_vectors), "build_vectors");

    for (int i = 0; i < num_vectors; i++) {
        print_distance(0, i); // Debug
    }

    int* labels = malloc(num_vectors * sizeof(int));
    double** centroids = malloc(K * sizeof(double*));

    // Allocate memory for vectors points and centroids
    for (int i = 0; i < K; i++) {
        centroids[i] = malloc(EXP_VECTOR_TABLE_SIZE * sizeof(double));
    }

    kmeans(vectors, num_vectors, labels, centroids);
    
    check(fflush(stdout), "fflush");

    // Print results
    printf("\nCluster Assignments:\n");
    for (int i = 0; i < num_vectors; i++) {
        printf("Point %d (in %d): %s\n", i, labels[i], strs[i]);
    }

    // Print results grouped by cluster
    printf("\nPoints By Cluster Assignment:\n");
    for (int cluster = 0; cluster <= K; cluster++) {
        printf("Cluster %d: ", cluster);
        for (int i = 0; i < num_vectors; i++) {
            if (labels[i] == cluster) {
                printf("%s,", strs[i]);
            }
        }
        printf("\n");
    }

    printf("\nFinal Centroids:\n");
    for (int j = 0; j < K; j++) {
        printf("Cluster %d: (", j);
        for (int dim = 0; dim < EXP_VECTOR_TABLE_SIZE; dim++) {
            printf("%.8lf", centroids[j][dim]);
            if (dim < EXP_VECTOR_TABLE_SIZE - 1) printf(", ");
        }
        printf(")\n");
    }
    
    printf("Done!\n");
    check(fflush(stdout), "fflush");

    return 0;
}
