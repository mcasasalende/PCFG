from django.shortcuts import render
from django.shortcuts import redirect
from django.conf import settings
from django.db.models import F, FloatField, Q
from django.db.models.functions import Cast
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.cache import never_cache
from django.urls import reverse
from string import digits as D  
from string import ascii_letters as L
from string import punctuation as S
from login.models import Rule
from login.models import TrainModel
from collections import defaultdict
import pandas as pd
import numpy as np
import timeit
import threading
import os
import random
from itertools import islice
L += "áéíóúÁÉÍÓÚüÜöÖäÄëËïÏàÀèÈìÌòÒùÙ"

# Create your views here.

def obtener_ficheros():
    ficheros_aux = os.listdir(settings.STATICFILES_DIRS[0])
    ficheros=[]
    for f in ficheros_aux:
        ficheros.append(str(f).removesuffix('.txt'))
    ficheros.remove("css")
    ficheros.remove("js")
    print(ficheros)

    return ficheros

@never_cache
def index(request):
    """
    Función vista para la página inicio del sitio.
    """
    # Genera contadores de algunos de los objetos principales
   
    # Libros disponibles (status = 'a')
    
    
    # Renderiza la plantilla HTML index.html con los datos en la variable contexto
    return render(request,'index.html',{'ficheros':obtener_ficheros()})

        

def add_to_dict(df, sub_preterminal, terminal_string):
    if not sub_preterminal in df['Preterminal'].values:
        df = df._append({'Preterminal': sub_preterminal, 'Terminales': [terminal_string], 'Apariciones': [1]}, ignore_index=True)
    else:
        idx = df[df.Preterminal == sub_preterminal].index[0]
        if not terminal_string in df.Terminales[idx]:
            df.Terminales[idx].append(terminal_string)
            df.Apariciones[idx].append(1)
        else:
            terminal_idx = df.Terminales[idx].index(terminal_string)
            df.Apariciones[idx][terminal_idx] += 1

    return df



def generate_terminals(passwords,start, end, df):
    #print("Thread con numeros de " + str(start) + " hasta " + str(end))
    cnt=0
    for p in passwords[start:end]:
        #print("Analizando passwd nº " + str(cnt+start))
        cnt+=1
        if cnt%100 == 0:
            print(cnt)
        if p == '\n':
            continue
        preterminal = ''
        i = 0
        #print('-----------------Analizando el passwd : '+ p )
        # Para cada letra, comprobamos que tipo de estructure preterminal es y la añadimos a la global de la password
        while i < len(p):
            #print("Analizo "+ p[i])
            if p[i] == '\n':
                break

            # Si hay una subcadena de simbolos
            if(p[i] in S):
                cntS = 0
                terminal_string=''

                while  i < len(p) and p[i] in S: # Contamos la subcadena
                    terminal_string += p[i]
                    cntS +=1
                    i+=1
                    
                sub_preterminal = 'S'+ str(cntS)
                preterminal += sub_preterminal
                #print('terminal string = '+ terminal_string)

                df = add_to_dict(df, sub_preterminal, terminal_string)
            if not i < len(p):
                break
            
            # Si hay una subcadena de letras

            if(p[i] in L or p[i] == ' '):
                cntL = 0
                terminal_string=''

                while  i < len(p) and (p[i] in L or p[i] == ' '):
                    terminal_string += p[i]
                    cntL +=1
                    i+=1
                
                sub_preterminal = 'L'+ str(cntL)
                preterminal += sub_preterminal
                #print('terminal string = '+ terminal_string)

                df = add_to_dict(df, sub_preterminal, terminal_string)
            if not i < len(p):
                break
            # Si hay una subcadena de difitos
            if(p[i] in D):
                cntD = 0
                terminal_string=''
                while  i < len(p) and p[i] in D:
                    terminal_string += p[i]
                    cntD +=1
                    i+=1
                sub_preterminal = 'D'+ str(cntD)
                preterminal += sub_preterminal
                #print('terminal string = '+ terminal_  string)

                df = add_to_dict(df, sub_preterminal, terminal_string)
            else:
                i+=1
                continue
            


        df = add_to_dict(df,'s',preterminal)
        


    return df


def generate_rules(df, mod):
    # Indice 1 porque el 0 es el nuero de la tupla
    # 2º indice -> 0 = Terminal, 1 = Preterminales, 2 = Apariciones"
    contador = 0
    numPass = 0
    numRul = 0
    for row in df.iterrows():
        contador += 1
        print(f'regla numero {contador}')
        print(row[1][0])
        total = sum(row[1][2])
        for t in row[1][1]:
            index = row[1][1].index(t)
            r = Rule.objects.create(LHS=row[1][0],RHS=t, totalTer = row[1][2][index], totalPret = total, prob = round(100*(row[1][2][index])/total,2), modelo = mod)
            r.save()
            TrainModel.objects.filter(model=mod).update(numRules=F('numRules') + 1)
        tm = TrainModel.objects.get(model=mod)
        print(f'Numero de contraseña {tm.numRules+1}')
        print(f'Contrasea {r.LHS} -> {r.RHS}')
        


    return

def update_rule(df):
    
    for row in df.iterrows():
        preterminal = row[[1][0]]
        terminales = row[[1][1]]
        apariciones = row[[1][2]]

        total = sum(apariciones)
        if Rule.objects.filter(LHS=preterminal).exists():
            all_rules__equal_preterminal = Rule.objects.filter(LHS=preterminal)
            all_rules__equal_preterminal.update(totalPret=F('totalPret')+total)
            all_rules__equal_preterminal.update(totalPret=F('totalTer')+total)
            for t in terminales:
                index = terminales.index(t)
                if Rule.objects.filter(LHS=preterminal, RHS=t).exists():
                    equal_rule = Rule.objects.filter(LHS=preterminal, RHS=t)
                    equal_rule.update(totalPret=F('totalPret')+apariciones[index])

                else:

                    index = terminales.index(t)
                    r = Rule.objects.create(LHS=row[1][0],RHS=t, totalTer = row[1][2][index], totalPret = total, prob = 100*(row[1][2][index])/total)
                    r.save()
        else:
            print(row[1][0])
            
            for t in row[1][1]:
                index = row[1][1].index(t)
                r = Rule.objects.create(LHS=row[1][0],RHS=t, totalTer = row[1][2][index], totalPret = total, prob = 100*(row[1][2][index])/total)
                r.save()


def upload_file(request):
    print("CONTROL")
    if request.method == 'POST' and request.FILES['file']:
        file = request.FILES['file']
        file_path = os.path.join(settings.STATICFILES_DIRS[0], file.name)

        with open(file_path, 'wb') as destination:
            for chunk in file.chunks():
                destination.write(chunk)
    

    request.session['ficheros'] = obtener_ficheros()
    return redirect('index')

def loading_screen(request):
    return render(request, 'loading_screen.html')

def password_rules(request):
    start_time = timeit.default_timer()

    modelo = request.POST['opcion']

    num_threads = 5

    Rule.objects.filter(modelo = modelo).delete()
    TrainModel.objects.filter(model = modelo).delete()
    #Rule.objects.all().delete()

    #Inicializamos el dataframe
    df = pd.DataFrame(columns=['Preterminal', 'Terminales', 'Apariciones'])

    #with open('WebApp/myWeb/login/static/BiblePass_part01.txt') as f:
    #with open('WebApp/myWeb/login/static/prueba.txt') as f:
    #    passwords_global = f.readlines()
    archivo_static = os.path.join(settings.STATICFILES_DIRS[0], modelo+'.txt')

    with open(archivo_static) as f:
        passwords_global = f.readlines()
        num_pass = len(passwords_global)
        df = generate_terminals(passwords_global, 0, num_pass, df)

    TrainModel.objects.create(model = modelo, numPasswords = num_pass)

    time4 = timeit.default_timer() - start_time
    print("Tiempo ="+ str(time4))


    dfs = np.array_split(df, num_threads)
   
    # Threads de generación de reglas
    rules = Rule.objects.none()
    threads = []

    for i, frame in enumerate(dfs):
        t = threading.Thread(target=generate_rules, args=(frame,modelo))
        threads.append(t)

    # Iniciar los threads
    for t in threads:
        t.start()

    # Esperar a que los threads terminen
    for t in threads:
        t.join()

    #rules = generarte_rules(terminals)
    rules = Rule.objects.all().order_by('-LHS','-prob')

    end_time = timeit.default_timer()

    totalTime = end_time - start_time
    print('Tiempo de ejecución = ' + str(totalTime) + 'seconds')

    #return JsonResponse({'rules': list(rules.values())})
    #return render(request, 'rules.html', {'passwords': {}, 'rules': rules, 'ficheros':obtener_ficheros()})
    return redirect('index')

def to_preterminal_structure(p):
    
    preterminal = ""
    sub_preterminals = []
    sub_chains = []
    i=0
    while i < len(p):
        # Si hay una subcadena de simbolos
        if(p[i] in S):
            cntS = 0
            ch = ''
            while i < len(p) and p[i] in S : # Contamos la subcadena
                ch += p[i]
                cntS +=1
                i+=1
            
            sub_preterminal = 'S'+ str(cntS)
            preterminal += sub_preterminal
            sub_preterminals.append(sub_preterminal)
            sub_chains.append(ch)
        if not i < len(p):
            break
        # Si hay una subcadena de letras
        if(p[i] in L):
            cntL = 0
            ch = ''
            while i < len(p) and p[i] in L :
                ch += p[i]
                cntL +=1
                i+=1

            sub_preterminal = 'L'+ str(cntL)
            preterminal += sub_preterminal
            sub_preterminals.append(sub_preterminal)
            sub_chains.append(ch)
        if not i < len(p):
            break

        # Si hay una subcadena de difitos
        if(p[i] in D):
            ch = ''
            cntD = 0
            while  i < len(p) and p[i] in D :
                ch += p[i]
                cntD +=1
                i+=1
   
            sub_preterminal = 'D'+ str(cntD)
            preterminal += sub_preterminal
            sub_preterminals.append(sub_preterminal)
            sub_chains.append(ch)
    return preterminal, sub_preterminals, sub_chains

def score_lenght(len):
    score = 0
    # Entre 0 y 30, damos puntos por la longitud
    if(len > 13):
        score = 30
    else:
        if len > 10:
            score = 20
        elif len > 7:
            score = 10
    
    return score

def passwd_score(passwd, mod):

    print(f'---------------------------------Analizando el password {passwd}----------------------------------------')
    # Obtenemos las estructuras y subcadenas de la password
    preterminal, sub_preterminals, sub_chains = to_preterminal_structure(passwd)

    reducciones = []

    # Valor segun longitud (0-30)
    score = score_lenght(len(passwd))

    reduccion_por_longitud = 30 - score

    # Booleano que nos indica si la contraseña aparece tal cual en el modelo
    aparece = True

    num_structures = len(sub_preterminals)

    aparicion = 0

    # Buscamos las preterminales de inicio
    rulesS = Rule.objects.filter(LHS='s',RHS=preterminal, modelo = mod)

    if(rulesS.exists()):

        rule_start = rulesS.first()
        print('Probabilidad de estructura preterminal ' + str(rule_start.RHS) + ' = ' + str(rule_start.prob))

        # Restamos la probabilidad de aparición al score
        red = (100*rule_start.prob /len(passwd))
        reducciones.append(round(red,2))
        aparicion += red
        print('Reducción de score por ' + str(rule_start.RHS) + ' = ' + str(rule_start.prob) + " = " + str(red))
        rules= rulesS

    else:
        aparece = False
        rules = Rule.objects.none()

    # Buscamos las subcadenas en las reglas
    for i in range(len(sub_preterminals)):
        # Buscamos las reglas que coincidan con alguna sub estructura
        r = Rule.objects.filter(LHS=sub_preterminals[i], RHS=sub_chains[i], modelo = mod)
        if r.exists():
            # Si coincide y es un digito o símbolo, al haber pocos, hacemos que la reducción afecte menos ya que la probabilidad sera mayor
            if(sub_chains[i][0] in D or  sub_chains[i][0] in S):
                reduccion = (r.first().prob *  len(sub_chains[i]))/(len(passwd)-len(sub_chains[i])+1)
                reducciones.append(round(reduccion,2))
                aparicion += reduccion
            # Si es una letra debe afectar más, ya que la probabilidad será menor, además cuanto más largo sea la subcadena, afectará aún más de manera exponencial
            if(sub_chains[i][0] in L):
                reduccion = (r.first().prob * 2** len(sub_chains[i]))/(len(passwd)-len(sub_chains[i])+1)
                reducciones.append(round(reduccion,2))
                aparicion += reduccion

            print("Para la regla " + str(sub_preterminals[i]+"-> "+sub_chains[i]) + " se reduce el score en " + str(reduccion))
        else:
            aparece = False

        rules = rules.union(r)


    print("aparicion = "+ str(aparicion))

    if aparicion > 70:
        aparicion = 70
    
    if aparece == False:
        print("La reduccion por las reglas es "+ str(aparicion))
        score = score + (70 - aparicion)
        print("El score es " + str(score))
    else:
        score = 0
    print("Las reducciones segun las reglas son : " + str(reducciones))

    return round(score,2), rules, reducciones

def similar_passwd(passwd, similars):
    l = list(passwd)
    for i in range(len(l)):
        if l[i] in similars:
            if bool(random.randint(0, 1)):
                l[i] = random.choice(similars[l[i]])
    passwd = "".join(l)
    return passwd

def extended_password(passwd):
    random_int = random.randint(0, 999)
    new_passwd = passwd + str(random_int)
    return new_passwd

def analize_passwd(request):

    # Obtenemos las variables
    passwd = request.POST.get('my_passwd')

    modelo = request.POST['opcion']

    similars = {'s':['$','S'] , '$':['s','S'],'e': ['3','E'],'3':['e','E'],'o':['0','+'], 'l': ['1','!','I','/'], 'p':['?'], '?':['p','P'],'b':['v'],
                 'a': ['4'],'4':['a','A'],'I':['1','l']}
    
    score, rules, reduccion = passwd_score(passwd, modelo)
    recomendations = []
    scores_rec = []
    len_redu_rec = []
    rules_rec = []

    for i in range(3):
        pass_aux = similar_passwd(passwd, similars)
        sc , r_aux , redu_aux= passwd_score(pass_aux, modelo)
        cnt = 0

        while (pass_aux == passwd or sc < score or pass_aux in recomendations):
            if (cnt >= 5):
                pass_aux = extended_password(passwd)

                if (cnt>= 7):
                    pass_aux = extended_password(pass_aux)
                
            else:
                pass_aux = similar_passwd(passwd, similars)
            sc , r_aux, redu_aux = passwd_score(pass_aux, modelo)
            print(f'Score para {pass_aux} = {sc}')
            cnt+=1
            if (cnt==10):
                break

        
        recomendations.append(pass_aux)
        sc , r_aux, redu_aux= passwd_score(pass_aux, modelo)
        scores_rec.append(sc)
        rules_rec.append(r_aux.count())
        len_redu_rec.append(score_lenght(30-len(pass_aux)))

    print(f'Recomendaciones = {recomendations}')
    print("------------------Info del modelo "+modelo+"------------------------")
    trainModel = TrainModel.objects.filter(model = modelo).first()
    print(f'numero de passwords del modelo {trainModel.numPasswords}')
    print(f'numero de reglas del modelo {trainModel.numRules}')
    print(f'El item 2 de polla es {scores_rec}')
    print(f'El item 2 de polla es {len_redu_rec}')
    passwd_por_regla = "{:.2f}".format(trainModel.numPasswords/trainModel.numRules) 

    combined_passwd_score = zip(recomendations, scores_rec, rules_rec, len_redu_rec)

    combined_rules_reducciones = zip(rules, reduccion)
    

    return render(request,'analyze.html',{'rules': rules, 'rule_data':combined_rules_reducciones,'passwd': passwd,'score':score,'len_redu':(30-score_lenght(len(passwd))), 'data': combined_passwd_score, 
                                          'modelo':modelo, 'passwd_por_regla':passwd_por_regla, 'ficheros':obtener_ficheros()});


def compare(request):
    model1 = request.POST.get('model')
    score =  request.POST.get('score')
    print(f'score es  es {score}')
    len_redu = request.POST.get('len_redu')
    print(f'len redu es {len_redu}')
    num_rules =  request.POST.get('reglas')
    passwd = request.POST.get('my_passwd')
    model2 = request.POST.get('opcion')

    trainModel1 = TrainModel.objects.filter(model = model1).first()
    trainModel2 = TrainModel.objects.filter(model = model2).first()
    passwd_por_regla1 = "{:.2f}".format(trainModel1.numPasswords/trainModel1.numRules) 
    passwd_por_regla2 = "{:.2f}".format(trainModel2.numPasswords/trainModel2.numRules) 

    score2, rules_2, redu_2= passwd_score(passwd, model2)

    pass_model1 = trainModel1.numPasswords
    rules_model1 = trainModel1.numRules

    pass_model2 = trainModel2.numPasswords
    rules_model2 = trainModel2.numRules

    if(passwd_por_regla1<passwd_por_regla2):
        modelo_mas_prendizaje = trainModel2.model
    else:
        modelo_mas_prendizaje = trainModel1.model

    return render(request,'compare.html',{'passwd':passwd,'score':score,'len_redu1':len_redu,'rules_used1':num_rules, 'score2':score2, 'rules_used2':rules_2.count(),'mejor_modelo':modelo_mas_prendizaje
                                          ,'modelo1':model1, 'modelo2':model2, 'passwd_por_regla1':passwd_por_regla1,'passwd_por_regla2':passwd_por_regla2,
                                         'pass1':pass_model1,'pass2':pass_model2,'rules1':rules_model1, 'rules2':rules_model2, 'ficheros':obtener_ficheros()})

def show_rules(request):
    mod = request.POST.get('opcion')
    if (mod ==None):
        rules = Rule.objects.all().order_by('-LHS','-prob')
    else:
        if(Rule.objects.filter(modelo = mod).exists()):
            rules = Rule.objects.filter(modelo = mod).order_by('-LHS','-prob')
        else:
            rules = []
    return render(request, 'rules.html', { 'rules': rules, 'modelo':mod, 'ficheros':obtener_ficheros()})

def passwd(request):
    return render(request,'passwd.html',context={})
    