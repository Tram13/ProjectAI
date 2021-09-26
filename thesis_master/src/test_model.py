import gensim
import re

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim.models.doc2vec import TaggedDocument

tokenizer = RegexpTokenizer(r'\w+')
nl_stop = get_stop_words('dutch')
p_stemmer = PorterStemmer()


def do_Doc2Vec_test():
    model = gensim.models.Doc2Vec.load('models/doc2vec/2005/200501/doc2vec_model')

    test_string = 'College van Beroep voor het bedrijfsleven Enkelvoudige kamer voor spoedeisende zaken AWB 04/1012 en ' \
                  '04/1013			1 februari 2005 11245 Gezondheids- en welzijnswet voor dieren             Besluit ' \
                  'biotechnologie bij dieren   Uitspraak in de zaak van: Vereniging AVS Proefdiervrij, ' \
                  'te \'s-Gravenhage, verzoekster, gemachtigde: mr. V.R. Wösten, juridisch adviseur te Amsterdam, ' \
                  'tegen de Minister van Landbouw, Natuur en Voedselkwaliteit, verweerder, gemachtigde: mr. K.J. Oost, ' \
                  'werkzaam bij verweerder, aan dit geding neemt voorts als partij deel: het Universitair Medisch ' \
                  'Centrum Utrecht,  gemachtigde: mr. M.G. Roessingh, werkzaam bij de Staf Juridische Zaken van het ' \
                  'Universitair Medisch Centrum Utrecht.   1.	De procedure Verzoekster heeft bij brief van 30 ' \
                  'november 2005, door het College ontvangen op 1 december 2005, een beroepschrift ingediend, ' \
                  'gericht tegen het besluit van verweerder van 22 september 2005, dat op 21 oktober 2005 - onder meer ' \
                  'in de Staatscourant - bekend is gemaakt. Het beroep is geregistreerd onder nummer Awb 04/1012. Bij ' \
                  'voormeld besluit heeft verweerder aan Universitair Medisch Centrum Utrecht (hierna: UMC Utrecht) ' \
                  'een vergunning verleend als bedoeld in artikel 66, eerste lid, van de Gezondheids- en welzijnswet ' \
                  'voor dieren (hierna: Gwd). Voorts heeft verzoekster bij brief van 30 november 2005, ' \
                  'door het College eveneens ontvangen op 1 december 2005, een verzoekschrift ingediend tot het ' \
                  'treffen van een voorlopige voorziening, ertoe strekkende dat het besluit van verweerder van  22 ' \
                  'september 2005 wordt geschorst. Dit verzoek is geregistreerd onder nummer Awb 04/1013. Bij brief ' \
                  'van 20 december 2005 heeft verweerder een verweerschrift ingediend. Bij brief van 22 januari 2005 ' \
                  'heeft verzoekster een reactie ingediend op het verweerschrift. Het onderzoek ter zitting heeft ' \
                  'plaatsgevonden op 25 januari 2005, alwaar partijen zijn verschenen bij hun gemachtigde. Voorts zijn ' \
                  'ter zitting verschenen A, werkzaam bij verzoekster, drs. R. Tramper, adjunct-secretaris van de ' \
                  'Commissie Biotechnologie bij dieren (hierna: Cbd), en B, werkzaam bij het UMC Utrecht.  2.	De ' \
                  'grondslag van het geschil 2.1	In de Gwd is voorzover van belang het volgende bepaald. "Artikel ' \
                  '66 1. Het is zonder vergunning verboden: a. het genetisch materiaal van dieren te wijzigen op een ' \
                  'wijze die voorbij gaat aan de natuurlijke barrières van geslachtelijke voortplanting en van ' \
                  'recombinatie; b. biotechnologische technieken bij een dier of een embryo toe te passen. 2. Op een ' \
                  'aanvraag voor een vergunning als bedoeld in het eerste lid beslist Onze Minister, gehoord de ' \
                  'Commissie biotechnologie bij dieren, bedoeld in artikel 69. 3. Een vergunning als bedoeld in het ' \
                  'eerste lid wordt slechts verleend indien naar het oordeel van Onze Minister: a. de handelingen geen ' \
                  'onaanvaardbare gevolgen hebben voor de gezondheid of het welzijn van dieren en  b. tegen de ' \
                  'handelingen geen ethische bezwaren bestaan. 4. In de vergunning wordt bepaald voor welke ' \
                  'handelingen zij is bedoeld. 5. Aan een vergunning kunnen voorschriften worden verbonden. Een ' \
                  'vergunning kan onder beperkingen worden verleend. Artikel 67 (…) 2. Onze Minister stelt regelen ' \
                  'omtrent het indienen van een aanvraag en omtrent de behandeling daarvan. Daarbij kan onder meer ' \
                  'worden bepaald: a. welke gegevens en bescheiden moeten worden overgelegd alvorens een aanvraag in ' \
                  'behandeling kan worden genomen; (…)" In de Regeling vergunning biotechnologie bij dieren (hierna: ' \
                  'de Regeling) is - voorzover relevant - het volgende bepaald. “Artikel 3 1. Een aanvraag voor een ' \
                  'vergunning bevat tenminste de volgende informatie: a. een uiteenzetting van de doelstellingen van ' \
                  'de biotechnologische handelingen, zowel op korte als lange termijn; b. een beschrijving van de toe ' \
                  'te passen technieken, van de uit te voeren handelingen en het belang daarvan in wetenschappelijk en ' \
                  'maatschappelijk opzicht, alsmede van de te gebruiken genen; c. de soorten en aantallen dieren; d. ' \
                  'een verantwoording van de gekozen aanpak zoals aangegeven in de onderdelen b en c in relatie tot de ' \
                  'in onderdeel a gegeven doelstellingen; e. een beschrijving van de voorzieningen voor de dieren en ' \
                  'hun bestemming na afloop van het onderzoek; f. een inschattnig van verwachte positieve en negatieve ' \
                  'effecten van de biotechnologische handelingen op de gezondheid, het welzijn en het functioneren van ' \
                  'alle dieren; g. een beschrijving van eventuele alternatieven voor de biotechnologische ' \
                  'handelingen;" 2.2	Bij de beoordeling van het verzoek om voorlopige voorziening gaat de ' \
                  'voorzieningenrechter uit van de volgende feiten en omstandigheden. - Het UMC Utrecht heeft onder ' \
                  'dagtekening 13 januari 2005 een aanvraag ingediend om verlening van een vergunning als bedoeld in ' \
                  'artikel 66, eerste lid, Gwd. Blijkens de aanvraag heeft het onderzoek als titel "Het ontstaan en de ' \
                  'behandeling van eetstoornissen: de rol van neuropeptiderge systemen".  - Verweerder heeft terzake ' \
                  'van de aanvraag advies gevraagd aan de Cbd. - Bij brieven van 2 februari 2005 en 1 maart 2005 heeft ' \
                  'het Cbd, via verweerder, UMC Utrecht verzocht om nadere informatie, die het UMC Utrecht bij brieven ' \
                  'van  9 februari 2005 en 4 maart 2005 heeft verstrekt.  - Bij brief van 29 maart 2005 heeft het Cbd ' \
                  'aan verweerder advies uitgebracht. Het Cbd heeft verweerder geadviseerd vergunning te verlenen ' \
                  'onder een aantal in het advies geformuleerde voorschriften en beperkingen.  - Op 14 mei 2005 heeft ' \
                  'verweerder een ontwerpbesluit, strekkende tot vergunningverlening, genomen en vervolgens ter inzage ' \
                  'gelegd. - Tijdens de hoorzitting van 15 juni 2005 is onder meer verzoekster terzake van mondelinge ' \
                  'bedenkingen gehoord en bij brief van 29 juni 2005 heeft verzoekster haar schriftelijke bedenkingen ' \
                  'tegen het ontwerpbesluit ingediend.   - Bij brief van 30 juli 2005 heeft het Cbd een reactie ' \
                  'gegeven op deze bedenkingen.  - Vervolgens heeft verweerder het bestreden besluit genomen. 3.	' \
                  'Het bestreden besluit In het bestreden besluit is onder meer het volgende overwogen: "1. Een ' \
                  'vergunning als bedoeld in artikel 66, eerste lid, onderdelen a en b, van de Gezondheids- en ' \
                  'welzijnswet voor dieren wordt verleend aan Universitair Medisch Centrum te Utrecht. 2. De ' \
                  'vergunning wordt verleend voor de werkzaamheden omschreven in beperking 2 en zoals omschreven in de ' \
                  'aanvraag van d.d. 13 januari 2005 met de aanvullingen hierop van d.d. 9 februari 2005 en d.d. 4 ' \
                  'maart 2005 van het Universitair Medisch Centrum te Utrecht met inachtneming van de in deze ' \
                  'vergunning opgenomen voorschriften en beperkingen. (…) Beperking 2 1. De onderhavige vergunning ' \
                  'heeft uitsluitend betrekking op het navolgende, zoals beschreven in de aanvraag van d.d. 13 januari ' \
                  '2005 met de aanvullingen hierop van d.d. 9 februari 2005 en d.d. 4 maart 2005 van Universitair ' \
                  'Medisch Centrum te Utrecht: a) het vervaardigen van genetisch gemodificeerde muizen door ' \
                  'micro-injectie van lentivirale deeltjes met siRNA\'s in de perivitelline ruimte van een bevruchte ' \
                  'eicel; b) daarbij wordt gebruik gemaakt van genconstructen die zijn samengesteld uit delen ' \
                  'gebaseerd op: ?  melanocortine receptoren, te weten MC-3 en MC-4; ?    genconstructen uit het ' \
                  'moleculaire standaardinstrumentarium (zie bijlage I bij het advies van de Commissie); 2. waarbij in ' \
                  'het kader van deze vergunning bij de biotechnologische handelingen in totaal maximaal 240 muizen ' \
                  'gebruikt mogen worden voor het genereren van 2 knock-down lijnen. 3. de biotechnologische ' \
                  'handelingen bij dieren dienen binnen 18 maanden na dagtekening van het besluit te zijn verricht. (' \
                  '…) Voorschrift 6 Vergunninghouder dient na afloop van de 18 maanden verslag te doen van de ' \
                  'bevindingen bij het genetisch modificeren van muizen met behulp van lentivirale deeltjes met ' \
                  'siRNA\'s." 4.	Het standpunt van verzoekster In het verzoekschrift en ter zitting heeft ' \
                  'verzoekster - samengevat - het volgende naar voren gebracht. 4.1 De grondslag van de aanvraag is ' \
                  'verlaten omdat verweerder feitelijk slechts een zeer klein deel van het gevraagde heeft vergund. De ' \
                  'vergunning is immers gevraagd voor een onderzoek van vijf jaar bij zowel muizen als ratten naar de ' \
                  'rol van diverse receptoren bij het ontstaan en de behandeling van eetstoornissen, terwijl de ' \
                  'vergunning is verleend voor het gedurende anderhalf jaar uitproberen van een nieuwe ' \
                  'modificatietechniek bij muizen. Dit brengt naar de opvatting van verzoekster mee dat verweerder bij ' \
                  'de totstandkoming van het bestreden besluit niet over voldoende specifieke gegevens beschikte ' \
                  'aangaande de betrokken belangen zoals genoemd in de Regeling.  4.2 Er ontbreekt een specificatie ' \
                  'van de doelstelling(en) op korte en lange termijn, terwijl de genoemde doelstellingen bovendien van ' \
                  'toepassing zijn op elk biotechnologisch onderzoek. Zwaarwegende belangen waarvoor het belang van de ' \
                  'betrokken dieren in het kader van het “nee tenzij”-beginsel zouden moeten wijken, zijn onvoldoende ' \
                  'aangetoond. Nu het thans vergunde betrekking heeft op een nieuwe techniek voor het genereren van ' \
                  'proefdieren, valt naar de opvatting van verzoekster niet in te zien waarom daarvoor de feitelijk ' \
                  'vergunde genen MC-3 en MC-4 nodig zijn, terwijl de schadelijke gevolgen van die genen en daarmee de ' \
                  'omvang van de bezwaren voor de proefdieren niet inzichtelijk zijn. 4.3 Het maatschappelijk belang ' \
                  'wordt onvoldoende benoemd om dit aspect in de besluitvorming mee te laten wegen. Voorts wordt een ' \
                  'aantal elementen van de schade bij de dieren genoemd zonder de omvang daarvan nader vast te ' \
                  'stellen. De vraag naar alternatieven is vanuit een te beperkt perspectief beantwoord, nu - slechts ' \
                  '- is gekeken naar eventuele alternatieven voor het voorgenomen onderzoek met transgene dieren en ' \
                  'niet voor alternatieve vormen van hulp aan patiënten met eetstoornissen. In dit verband verwijst ' \
                  'verzoekster naar het door haar bij het beroepschrift overgelegde advies van de Gezondheidsraad, ' \
                  'waarin wordt gesteld dat genetische factoren weliswaar een rol spelen bij het ontstaan van ' \
                  'overgewicht, maar dat de invloed van omgevingsfactoren van doorslaggevende betekenis lijkt te zijn. ' \
                  '   4.4 Tot slot is de strekking van voorschrift 6 onvoldoende bepaald. Niet duidelijk is immers aan ' \
                  'wie verslag moet worden uitgebracht. Hoewel tegen dit - reeds in het ontwerpbesluit opgenomen - ' \
                  'voorschrift geen bedenking is aangevoerd, kan deze grond in het kader van de onderhavige procedure(' \
                  's) wel worden beoordeeld. Er bestaat geen wettelijke grondslag voor het ongegrond verklaren van een ' \
                  'beroepsgrond uitsluitend vanwege de omstandigheid dat deze niet eerder als bedenking in de daarvoor ' \
                  'bestemde fase naar voren is gebracht. Subsidiair stelt verzoekster dat gelet op de bijzondere ' \
                  'plaats die algemene rechtsbeginselen binnen het bestuursrecht innemen, daarmee samenhangende ' \
                  'beroepsgronden bezwaarlijk kunnen worden gepasseerd op de grond dat deze niet als bedenking zijn ' \
                  'ingebracht. 5.	De beoordeling van het geschil 5.1	Ingevolge artikel 8:81, van de Algemene wet ' \
                  'bestuursrecht (hierna: Awb) juncto artikel 19, eerste lid, van de Wet bestuursrechtspraak ' \
                  'bedrijfsorganisatie (hierna: Wbb), kan hangende het beroep bij het College, de voorzieningenrechter ' \
                  'van het College een voorlopige voorziening treffen, indien onverwijlde spoed, gelet op de betrokken ' \
                  'belangen, dat vereist. Ingevolge artikel 8:86, eerste lid, Awb juncto artikel 19, eerste lid, ' \
                  'Wbb kan, indien beroep bij het College is ingesteld en de voorzieningenrechter van oordeel is dat ' \
                  'na de zitting nader onderzoek redelijkerwijs niet kan bijdragen aan de beoordeling van de zaak, ' \
                  'de voorzieningenrechter onmiddellijk uitspraak doen in de hoofdzaak, mits procespartijen hiervoor ' \
                  'toestemming hebben gegeven. Ter zitting hebben partijen deze toestemming verleend. Op grond van de ' \
                  'door partijen overgelegde stukken en het verhandelde ter zitting is de voorzieningenrechter van ' \
                  'oordeel dat onmiddellijke uitspraak in de hoofdzaak kan worden gedaan, waartoe als volgt wordt ' \
                  'overwogen. 5.2	Centraal staat de beantwoording van de vraag of verweerder bij het bestreden ' \
                  'besluit tot verlening van de vergunning als bedoeld in artikel 66, eerste lid, Gwd heeft kunnen ' \
                  'besluiten. Aan verzoekster moet worden toegegeven dat het onderzoek, zoals dat in de aanvraag ' \
                  'uiteen is gezet, van - aanmerkelijk - grotere omvang en duur is dan hetgeen bij het bestreden ' \
                  'besluit aan UMC Utrecht is vergund. Anders dan verzoekster stelt brengt dit naar het oordeel van de ' \
                  'voorzieningenrechter echter niet mee dat de band tussen de aanvraag en datgene dat vergund is niet ' \
                  '- meer - zou bestaan, of - zoals verzoekster het stelt - dat de grondslag van de aanvraag zou zijn ' \
                  'verlaten. Uiteindelijk doel van het aangevraagde is, zo blijkt uit de gedingstukken, om door middel ' \
                  'van in vivo onderzoek inzicht te verwerven in de rol die neuropeptide receptoren spelen in de ' \
                  'energiebalans, meer in het bijzonder in het ontstaan van eetstoornissen zoals obesitas en anorexia. ' \
                  'Het UMC Utrecht heeft dienaangaande in de aanvraag uiteengezet dat de resultaten van deze - ' \
                  'fundamenteel wetenschappelijke - hoofddoelstelling een bijdrage kunnen leveren aan de gezondheid ' \
                  'voor de mens; soms direct door het ontstaan van een fenotype dat vergelijkbaar is met een ' \
                  'aangeboren of erfelijke afwijking bij de mens en soms indirect omdat de genetisch gemodificeerde ' \
                  'dieren modellen opleveren om nieuwe geneesmiddelen te ontwikkelen. Als maatschappelijk belang van ' \
                  'het onderzoek heeft het UMC Utrecht vermeld dat bestrijding van op steeds grotere schaal ' \
                  'voorkomende eetstoornissen (anorexia en obesitas) vanuit die optiek van belang is en dat er (nog) ' \
                  'geen goede geneesmiddelen zijn ter bestrijding van die stoornissen, terwijl duidelijk is dat de ' \
                  'hersenen een belangrijke rol spelen in de regulatie van de energiebalans. In haar reactie van 9 ' \
                  'februari 2005 heeft het UMC Utrecht erkend dat culturele en omgevingsfactoren een belangrijke rol ' \
                  'spelen bij eetstoornissen, doch zij heeft hierbij aangetekend dat de perceptie van deze factoren ' \
                  'via de hersenen tot stand komt en dat neuropeptiden hierbij een essentiële rol spelen. Ten aanzien ' \
                  'van de keuze voor de proefdieren is in de aanvraag uiteengezet dat de keuze voor de muis is ' \
                  'ingegeven door het feit dat de technieken voor genetische modificatie voor dit dier het best zijn ' \
                  'ontwikkeld. Voorts is aangegeven dat de efficiëncy om transgene ratten te maken recentelijk ' \
                  'aanzienlijk is verhoogd door eencellige embryo’s via injectie van lentivirale partikels te ' \
                  'infecteren met het transgen (de door UMC Utrecht aangevraagde techniek) en dat de rat (met name ' \
                  'vanwege zijn grootte) in sommige gevallen een beter proefdier is dan de muis. 5.3 	Uit de stukken ' \
                  'en het verhandelde ter zitting blijkt dat op advies van de Cbd de vergunning is verleend voor het ' \
                  'eerste deel van het in de aanvraag uiteengezette onderzoek, te weten het uitproberen van de nieuwe ' \
                  'techniek voor het vervaardigen van genetisch gemodificeerde muizen door micro-injectie van ' \
                  'lentivirale deeltjes met siRNA\'s in de perivitelline ruimte van een bevruchte eicel. Achtergrond ' \
                  'van de door de Cbd geadviseerde beperking van het thans vergunde onderzoek is blijkens de brief van ' \
                  'de Cbd van 1 maart 2005 - kort gezegd - dat nog onvoldoende duidelijk is of de door UMC Utrecht te ' \
                  'hanteren, in Nederland nieuwe, techniek voor het maken van de gewenste genetisch gemodificeerde ' \
                  'dieren (ook ratten) werkt en of daarmee inderdaad kan worden bereikt dat minder proefdieren nodig ' \
                  'zijn. Om die reden heeft de Cbd verweerder geadviseerd UMC Utrecht in eerste instantie vergunning ' \
                  'te verlenen voor een beperkte periode en voor een beperkt aantal muizenlijnen teneinde UMC Utrecht ' \
                  'in de gelegenheid te stellen met betrekking tot de te gebruiken techniek te komen tot een zogenoemd ' \
                  '“proof of principle”, welk advies door verweerder bij het bestreden besluit is overgenomen. 5.4 	' \
                  'Aldus beschouwd is het vergunde onderzoek een onderzoek naar de haalbaarheid van het bij de ' \
                  'aanvraag uiteengezette onderzoek op lange(re) termijn, hetgeen meebrengt dat pas tot eventuele ' \
                  'vergunningverlening voor vervolgonderzoek wordt overgegaan nadat de resultaten van het ' \
                  'haalbaarheidsonderzoek bekend zullen zijn. In dit kader is naar het oordeel van de ' \
                  'voorzieningenrechter ook het vergunningvoorschrift 6, waarover hierna meer, te plaatsen. Anders dan ' \
                  'verzoekster stelt is dan ook geen sprake van een weigering van een vergunning voor het meerdere dat ' \
                  'UMC Utrecht heeft aangevraagd, doch veeleer van een opschorting van besluitvorming dienaangaande in ' \
                  'afwachting van de resultaten van het thans vergunde onderzoek. 5.5 	Het vorenstaande brengt mee ' \
                  'dat toetsing van de verleende vergunning dient plaats te vinden tegen de achtergrond van het gehele ' \
                  'onderzoek, zoals dat is aangevraagd.  In het licht van de aldus te verrichten toetsing en de in het ' \
                  'kader van de aanvraagprocedure door UMC verschafte informatie kan de voorzieningenrechter ' \
                  'appellante niet volgen in haar stellingname dat verweerder bij de totstandkoming van het bestreden ' \
                  'besluit over onvoldoende gegevens beschikte.  Voorts is de voorzieningenrechter van oordeel dat, ' \
                  'bezien in het licht van het totale voorgenomen onderzoek, de korte en lange termijn doelstelling ' \
                  'door de aanvrager voldoende concreet zijn omschreven. Blijkens de gedingstukken moet als korte ' \
                  'termijn doelstelling worden gezien het maken van muizen (en als de techniek voldoet ratten) waarmee ' \
                  'de rol van neuropeptide receptoren bij de regulatie van de energiebalans kan worden onderzocht. Uit ' \
                  'bedoelde stukken volgt tevens dat het begrijpen van de rol van neuropeptide receptoren bij de ' \
                  'regulatie van de energiebalans en het, op basis van de aldus verworven kennis, ontwikkelen van ' \
                  'nieuwe geneesmiddelen die van belang kunnen zijn bij de behandeling van patiënten met ' \
                  'eetstoornissen als lange termijn doelstellingen kunnen worden beschouwd. Dit betreft het ' \
                  'uiteindelijke doel van het voorgenomen onderzoek. De grief van verzoekster dat een korte en lange ' \
                  'termijn doelstelling ontbreken, mist in het licht van het vorenstaande feitelijke grondslag. Voorts ' \
                  'heeft verweerder zich op het standpunt kunnen stellen dat de zwaarwegende belangen die zijn gemoeid ' \
                  'met onderhavig onderzoek door UMC Utrecht voldoende inzichtelijk zijn gemaakt. Bovendien heeft ' \
                  'verweerder zich, in navolging van de Cbd en in het licht van de hiervoor genoemde korte en lange ' \
                  'termijn doelstelling, in redelijkheid op het standpunt kunnen stellen dat het onderzoek zowel ' \
                  'wetenschappelijk als maatschappelijk van zodanig belang is, dat dit opweegt tegen de mogelijke ' \
                  'effecten op de gezondheid en het welzijn van de proefdieren en de aantasting van hun integriteit.  ' \
                  'In dit verband acht de voorzieningenrechter tevens van belang dat, zoals verzoekster terecht ' \
                  'aanvoert, de gevolgen voor de dieren van de door UMC Utrecht aangevraagde techniek niet met een ' \
                  'grote mate van zekerheid kunnen worden vastgesteld, doch dat hierin mede een reden is gelegen de ' \
                  'vergunning in omvang en duur te beperken.  Bij de beoordeling of wordt voldaan aan het “nee, ' \
                  'tenzij”-beginsel dient naar het oordeel van de voorzieningenrechter mede te worden gekeken naar de ' \
                  'lange termijndoelstelling, nu het met behulp van nieuwe technieken maken van proefdieren immers ' \
                  'geen doel in zich zelf is. Daargelaten de door verzoekster betwiste percentages van - mortaliteit ' \
                  'bij - anorexia, heeft zij niet betwist dat (de gevolgen van) eetstoornissen in toenemende mate een ' \
                  'ernstige bedreiging vormen van de volksgezondheid, terwijl uit het door haar overgelegde advies van ' \
                  'de Gezondheidsraad blijkt dat ook genetische factoren bij obesitas een rol spelen. Juist nu de ' \
                  'wijze waarop de aanvrager de proefdieren wil genereren betrekkelijk nieuw is, heeft verweerder in ' \
                  'navolging van de Cbd naar het oordeel van de voorzieningenrechter vanuit de door hem in acht te ' \
                  'nemen zorgvuldigheid op goede gronden besloten vooralsnog te volstaan met het vergunnen van ' \
                  'onderzoekshandelingen, waarmee met een beperkt aantal muizen de bruikbaarheid van deze methode bij ' \
                  'wege van “proof of principle” zou kunnen worden aangetoond. Indien uit de voorgeschreven evaluatie ' \
                  'blijkt dat de daarbij toe te passen techniek haalbaar is en zou blijken dat als gevolg van die ' \
                  'techniek minder proefdieren nodig zijn, kan op basis daarvan verdere beslutivorming plaatsvinden. ' \
                  'Wel merkt de voorzieningenrechter in dit verband ten overvloede op dat gelet op de inhoud van het  ' \
                  'bestreden besluit niet - zonder meer - voor de hand lijkt te liggen eventueel door het UMC Utrecht ' \
                  'te verrichten vervolgonderzoek toe te staan middels een wijziging van de thans verleende ' \
                  'vergunning. Nu dit echter geen onderdeel vormt van het bestreden besluit is dit voor de beoordeling ' \
                  'van het geschil niet van belang.   In aanmerking nemend hetgeen de Cbd omtrent de eenvoudig vast te ' \
                  'stellen effectiviteit van de genetische modificatie heeft overwogen en het doel van het ' \
                  'uiteindelijk beoogde onderzoek, heeft verweerder bij het bestreden besluit de daarbij genoemde ' \
                  'genconstructen op goede gronden kunnen vergunnen. De voorzieningenrechter kan verzoekster voorts ' \
                  'niet volgen in haar stelling dat een aantal elementen van de schade bij de dieren wordt genoemd, ' \
                  'zonder dat daarvan de omvang nader is vastgesteld. Blijkens de brief van UMC Utrecht van 9 februari ' \
                  '2005 en het advies van de Cbd is het verwachte gevolg van de thans vergunde genconstructen, ' \
                  'die de MC3 en MC4-receptoren “downreguleren”, dat bij de proefdieren opstapeling van vet en ' \
                  'overgewicht zal ontstaan. Juist de eenvoudige waarneembaarheid van deze verwachte gevolgen is voor ' \
                  'verweerder reden geweest het thans vergunde onderzoek tot deze genconstructen te beperken.  Tevens ' \
                  'is de voorzieningenrechter van oordeel dat verweerder zich op het standpunt heeft kunnen stellen ' \
                  'dat reële alternatieven niet voorhanden zijn. Hierbij wordt in aanmerking genomen dat verweerder ' \
                  'onbetwist heeft gesteld dat onderzoek in cel- en weefselkweken weliswaar waardevolle informatie ' \
                  'oplevert over de rol van neuropeptide receptoren maar dat de effecten op eetgedrag en vetopslag ' \
                  'uitsluitend kunnen worden bestudeerd in een geheel en relatief complex organisme als een zoogdier. ' \
                  'Voorts heeft verweerder aangegeven dat weliswaar therapieën bestaan voor de behandeling van de ' \
                  'eetstoornissen, doch dat aannemelijk is dat voor bepaalde groepen patiënten noch dieetaanpassingen, ' \
                  'noch begeleiding en psychotherapie effect hebben, zodat de geneesmiddelen die uiteindelijk uit het ' \
                  'beoogde onderzoek kunnen voortkomen een belangrijke farmacologische ondersteuning kunnen vormen bij ' \
                  'de reeds beschikbare behandelingsvormen. Met betrekking tot de grief van verzoekster dat ' \
                  'voorschrift 6 van de verleende vergunning onvoldoende bepaald is, stelt de voorzieningenrechter ' \
                  'voorop dat deze grief weliswaar niet in de bedenkingenfase naar voren is gebracht, maar dat deze ' \
                  'niet raakt aan de inhoud van het vergunde, doch uitsluitend aan een procedurevoorschrift met het ' \
                  'oog op eventueel vervolgonderzoek. Aldus valt niet in te zien waarom rechterlijke toetsing van dit ' \
                  'voorschrift verweerder in zijn (verdedigings)positie zou schaden.  Met betrekking tot de inhoud van ' \
                  'dit voorschrift merkt de voorzieningenrechter op dat dit gelet op het vergunde (' \
                  'haalbaarheids)onderzoek geen ander doel kan dienen dan een evaluatie, teneinde te beoordelen of - ' \
                  'ook - het door UMC Utrecht beoogde vervolgonderzoek voor vergunningverlening in aanmerking komt. De ' \
                  'voorzieningenrechter gaat er dan ook vanuit dat het voorgeschreven evaluatieverslag aan verweerder ' \
                  'moet worden uitgebracht en met het oog op advisering bij eventuele verdere besluitvorming aan de ' \
                  'Cbd zal worden voorgelegd, temeer nu het de Cbd is geweest die dit voorschrift heeft geadviseerd.  ' \
                  '5.6	Gelet op het vorenstaande komt de voorzieningenrechter tot de slotsom dat verweerder tot het ' \
                  'verlenen van de vergunning onder de daarbij opgenomen beperkingen en voorschriften heeft kunnen ' \
                  'besluiten.  Al het vorenwogene leidt tot het oordeel dat het beroep ongegrond dient te worden ' \
                  'verklaard. Gelet op de beslissing in de hoofdzaak bestaat voorts geen aanleiding voor het treffen ' \
                  'van een voorlopige voorziening. Het hiertoe strekkende verzoek dient dan ook te worden afgewezen. ' \
                  'Er is geen aanleiding voor een proceskostenveroordeling met toepassing van artikel 8:75 Awb. 6. De ' \
                  'beslissing De voorzieningenrechter: - verklaart het beroep ongegrond; - wijst het verzoek tot het ' \
                  'treffen van een voorlopige voorziening af. Aldus gewezen door mr. M.A. van der Ham, ' \
                  'in tegenwoordigheid van mr. P.M. Beishuizen als griffier, en uitgesproken in het openbaar op 1 ' \
                  'februari 2005. w.g. M.A. van der Ham				w.g. P.M. Beishuizen '

    raw = test_string.lower()
    raw = re.sub('\t', ' ', raw)
    raw = re.sub(' +', ' ', raw)
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in nl_stop]

    # remove numbers
    number_tokens = [re.sub(r'[\d]', ' ', i) for i in stopped_tokens]
    number_tokens = ' '.join(number_tokens).split()

    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in number_tokens]

    new_vector = model.infer_vector(stemmed_tokens)
    sims = model.docvecs.most_similar([new_vector], topn=10)

    print(sims)


do_Doc2Vec_test()
