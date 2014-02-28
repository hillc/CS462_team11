# -*- Mode: Python; indent-tabs-mode: nil -*-

# Please adhere to the PEP 8 style guide:
#     http://www.python.org/dev/peps/pep-0008/
import sys
from math import log
import sqlite3
import networkx as nx
import re
import bisect
import datetime
import iso8601
from nltk.stem import PorterStemmer
import cPickle
try:
    import matplotlib.pyplot as plt
except:
    raise
from util import *

# helps us to visualize what is happening with indents

import logging
logging.basicConfig(filename='callstack.log', filemode='w', level=logging.DEBUG)
logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

import inspect
import traceback
enable_callstack_printing = True
last_debug = ''
def debug(msg):
    global enable_callstack_printing
    global last_debug
    if not enable_callstack_printing:
        return

    indentation_level = len(traceback.extract_stack()) - 13
    output = '{i} [{m}] ({c})'.format(i='.' * indentation_level, m=msg, c=indentation_level)
    if output == last_debug:
        return # don't print out if it's just a repeat
    else:
        last_debug = output
    logger.debug(output)

stop = {}
sourcefile = ''
activation_root = ''


def stopwords():
    debug('stopwords')
    f = open('je.txt')
    for line in f:
        stop[line[0:-1]] = 1


if len(stop) == 0:
    stopwords()


def indexCamelWords(string):
    #debug('indexCamelWords')
    #debug([('word',
    #         PorterStemmer().stem(word).lower()) for word in \
    #            camelSplit(string) if word.lower() not in stop]);
    return [('word',
             PorterStemmer().stem(word).lower()) for word in \
                camelSplit(string) if word.lower() not in stop]


def indexWords(string):
    #debug('indexWords')
    #debug([('word',
    #         word.lower()) for word in re.split(r'\W+|\s+',string) \
    #           if word != '' and word.lower() not in stop])
    return [('word',
             word.lower()) for word in re.split(r'\W+|\s+',string) \
                if word != '' and word.lower() not in stop]


# Package (type -> package)
# Imports (type -> type)
# Extends (type -> type)
# Implements (type -> type)
# Method declaration (type -> method)
# Constructor invocation (method -> method)
# Method invocation (method -> method)
# Variable declaration (type -> variable)
# Variable type (variable -> type)


# TODO: This function is currently dead code; it may be broken
#def checkTopology(g):
#    '''For which graphs are there disconnects?'''
#    for key in g:
#        if not nx.is_connected(g[key]):
#            print key


# TODO: This function is currently dead code; it may be broken
#def graphDiameters(graphs):
#    '''What's the radius and diameter of each graph?'''
#    for key in graphs:
#        if len(graphs[key]) < 2000:
#            if len(nx.connected_component_subgraphs(graphs[key])) > 1:
#                for g in nx.connected_component_subgraphs(graphs[key]):
#                    print key, len(g), nx.radius(g), nx.diameter(g)
#            continue
#        print key, len(graphs[key]), nx.radius(graphs[key]), \
#            nx.diameter(graphs[key])


def loadScents(graphs={}):
    i = 0
    '''Load just the scent portion of the graphs'''
    debug('Begin loadScents')
    conn = sqlite3.connect(sourcefile)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('''select user,action,target,referrer,agent from logger_log where action in ('Package','Imports','Extends','Implements','Method declaration','Constructor invocation', 'Method invocation', 'Variable declaration', 'Variable type', 'Constructor invocation scent', 'Method declaration scent', 'Method invocation scent')''')
    for row in c:
        user, action, target, referrer, agent = (row['user'][0:3],
                                                 row['action'],
                                                 fix(row['target']),
                                                 fix(row['referrer']),
                                                 row['agent'])
        if agent not in graphs:
            graphs[agent] = nx.Graph()
        if action in ('Package', 'Imports', 'Extends', 'Implements',
                      'Method declaration', 'Constructor invocation',
                      'Method invocation', 'Variable declaration',
                      'Variable type'):
            # Connect class to constituent words
            for word in indexWords(target):
                graphs[agent].add_edge(target, word)
            # Connect import to constituent words
            for word in indexWords(referrer):
                graphs[agent].add_edge(referrer, word)
        elif action in ('Constructor invocation scent',
                        'Method declaration scent',
                        'Method invocation scent'):
            for word in indexWords(referrer):
                graphs[agent].add_edge(target,word)
            for word in indexCamelWords(referrer):
                graphs[agent].add_edge(target,word)
        i = i+1
        edges = [(u,v) for (u,v,d) in graphs[agent].edges(data=True)]
        pos = nx.spring_layout(graphs[agent])
        nx.draw_networkx_nodes(graphs[agent],pos,node_size=700)
        nx.draw_networkx_edges(graphs[agent], pos, edgelist=edges, width=6)
        nx.draw_networkx_labels(graphs[agent],pos,font_size=20,font_family='sans-serif')
        plt.axis('off')
        savename = 'graph_' +str(i)+ '.png'
        plt.savefig(savename)
        plt.show
    c.close()
    #debug(graphs)
    debug('End loadScents')
    return graphs


def loadTopology(graphs={}):
    '''Load just the topology portion of the graphs'''
    debug('Begin loadTopology')
    conn = sqlite3.connect(sourcefile)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('''select user,action,target,referrer,agent from logger_log where action in ('Package','Imports','Extends','Implements','Method declaration','Constructor invocation', 'Method invocation', 'Variable declaration', 'Variable type')''')
    for row in c:
        user, action, target, referrer, agent = (row['user'][0:3],
                                                 row['action'],
                                                 fix(row['target']),
                                                 fix(row['referrer']),
                                                 row['agent'])
        if agent not in graphs:
            graphs[agent] = nx.Graph()
        # Connect topology
        ntarget = normalize(target)
        nreferrer = normalize(referrer)
        pack = package(target)
        proj = project(target)
        if proj != '':
            graphs[agent].add_edge('packages', proj)
        if pack != '':
            graphs[agent].add_edge('packages', pack)
        if pack != '' and ntarget != '':
            graphs[agent].add_edge(pack, ntarget)
        if pack != '' and proj != '':
            graphs[agent].add_edge(proj, pack)
        graphs[agent].add_edge(target, referrer)
        if ntarget != '':
            graphs[agent].add_edge(ntarget, target)
        if nreferrer != '':
            graphs[agent].add_edge(nreferrer, referrer)
        if ntarget != '' and nreferrer != '':
            graphs[agent].add_edge(ntarget, nreferrer)
    c.close()
    debug('End loadTopology')
    return graphs


def loadScrolls(graphs={}):
    '''Load the scroll (adjacency) information from the db'''
    debug('Begin loadScrolls')
    offsets = {}

    def sorted_insert(l, item):
        debug('sorted_insert')
        l.insert(bisect.bisect_left(l, item), item)

    def add_offset(agent, timestamp, loc, target, referrer):
        '''Update method declaration offset in offsets data structure'''
        debug('add_offset')
        if agent not in offsets:
            offsets[agent] = {}
        if loc not in offsets[agent]:
            offsets[agent][loc] = []
        if agent not in graphs:
            graphs[agent] = nx.Graph()
        # Remove any existing occurrence of given target
        for item in offsets[agent][loc]:
            if item['target'] == target:
                offsets[agent][loc].remove(item)
        # Maintain an ordered list of method declaration offsets that is
        # always current as of this timestamp.
        sorted_insert(offsets[agent][loc],
                      {'referrer': referrer, 'target': target})
        for i in range(len(offsets[agent][loc])):
            if i+1 < len(offsets[agent][loc]):
                graphs[agent].add_edge(offsets[agent][loc][i]['target'],
                                       offsets[agent][loc][i+1]['target'])

    conn = sqlite3.connect(sourcefile)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("select timestamp,action,target,referrer,agent from logger_log where action in ('Method declaration offset') order by timestamp")
    for row in c:
        timestamp, action, agent, target, referrer = \
            (iso8601.parse_date(row['timestamp']),
             row['action'],
             row['agent'],
             row['target'],
             int(row['referrer']))
        loc = normalize(target)
        if loc:
            add_offset(agent, timestamp, loc, target, referrer)
    c.close()
    debug('End loadScrolls')
    return graphs


def loadPaths():
    '''
    Reconstruct the original method-level path through the source code by
    matching text selection offsets to method declaration offsets. Text
    selection offsets occur later in the data than method declaration
    offsets, so we can use this knowledge to speed up the reconstruction
    query. It returns: A timestamped sequence of navigation among methods
    (or classes if no method declaration offsets are available for that
    class)
    '''
    debug('Begin loadPaths')
    nav = {}
    out = {}
    offsets = {}

    def sorted_insert(l, item):
        debug('sorted_insert')
        l.insert(bisect.bisect_left(l, item), item)

    def add_nav(agent, timestamp, loc, referrer):
        '''Add text selection navigation events to a structure'''
        debug('add_nav')
        # Add 20 seconds to each nav to work around a race condition in PFIG
        timestamp += datetime.timedelta(seconds=20)
        if agent not in nav:
            nav[agent] = []
        nav[agent].append({'timestamp': timestamp,
                           'referrer': referrer,
                           'loc': loc})

    def add_offset(agent, timestamp, loc, target, referrer):
        '''Update method declaration offset in offsets data structure'''
        debug('add_offset')
        loc2 = loc.split('$')[0]
        if agent not in offsets:
            offsets[agent] = {}
        if loc2 not in offsets[agent]:
            offsets[agent][loc2] = []
        # Remove any existing occurrence of given target
        for item in offsets[agent][loc2]:
            if item['target'] == target:
                offsets[agent][loc2].remove(item)
        # Maintain an ordered list of method declaration offsets that is
        # always current as of this timestamp.
        sorted_insert(offsets[agent][loc2],
                      {'referrer': referrer, 'target': target})

    def get_out(agent, timestamp):
        '''Match text selection navigation data to the method declaration
        offset data to determine the method being investigated
        '''
        debug('get_out')
        if agent not in out:
            out[agent] = []
        while agent in nav and agent in offsets and \
                len(nav[agent]) > 0 and \
                timestamp >= nav[agent][0]['timestamp']:
            referrer = nav[agent][0]['referrer']
            loc = nav[agent][0]['loc']
            curts = nav[agent][0]['timestamp']
            # If method declaration offsets are unavailable, append to out
            # without the method name
            if loc not in offsets[agent]:
                out[agent].append({'target': referrer,
                                   'timestamp': curts})
                nav[agent].pop(0)
                continue
            # Otherwise, lookup the method declaration offset for this
            # navigation event
            # TODO: Is the 'zz' supposed to be max string? If so, there must be
            # a better way.
            index = bisect.bisect_left(offsets[agent][loc],
                                       {'referrer': referrer,
                                        'target': 'zz'}) - 1
            if index < 0:
                index = 0
            out[agent].append({'target': offsets[agent][loc][index]['target'],
                               'timestamp': curts})
            nav[agent].pop(0)

    conn = sqlite3.connect(sourcefile)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("select timestamp,action,target,referrer,agent from logger_log where action in ('Method declaration offset','Text selection offset') order by timestamp")
    t = None
    for row in c:
        timestamp, action, agent, target, referrer = \
            (iso8601.parse_date(row['timestamp']),
             row['action'],
             row['agent'],
             row['target'],
             int(row['referrer']))
        loc = normalize(target)
        if loc:
            if action == 'Text selection offset' and referrer > 0:
                add_nav(agent, timestamp, loc, referrer)
            elif action == 'Method declaration offset':
                add_offset(agent, timestamp, loc, target, referrer)
            get_out(agent, timestamp)
            t = timestamp
    c.close()
    for agent in nav:
        if len(nav[agent]) > 0:
            get_out(agent, t)
            for item in nav[agent]:
                out[agent].append({'target': item['loc'], 'timestamp': t})
    debug('End loadPaths')
    return out


def init():
    debug('init');
    g = {}
    print 'Loading scent data...'
    loadScents(g)
    print 'Loading topology data...'
    loadTopology(g)
    print 'Loading scroll data...'
    loadScrolls(g)
    print 'Loading paths...'
    nav = loadPaths()
    return nav, g


def between_method(a, b):
    debug('between_method')
    return a != b


def between_class(a, b):
    debug('between_class')
    return normalize(a) != normalize(b)


def between_package(a, b):
    debug('between_package')
    return package(a) != package(b)


def pathsThruGraphs(nav, graph, func, level=between_method,
                    data={}, words=[], agents=[]):
    '''Compute information about traversals through graphs.'''
    debug('Begin pathsThruGraphs')
    if len(agents) == 0:
        agents = [agent for agent in nav]
    for agent in agents:
        if len(nav[agent]) == 0:
            continue
        if agent not in graph:
            continue
        pprev = None
        prev = nav[agent][0]
        i = 0
        j = 0
        for row in nav[agent]:
            j += 1
            if level(row['target'], prev['target']):
                i += 1
                func(data, agent, graph[agent], pprev, prev, row, i, words)
                pprev = prev
            prev = row
    debug('End pathsThruGraphs')
    return data


def backtracks(data, agent, graph, pprev, prev, row, i, words):
    '''Pass to pathsThruGraphs: record A -> B -> A navigation at some level'''
    debug('backtracks')
    if agent not in data:
        data[agent] = 0
    if pprev != None and pprev['target'] == row['target']:
        data[agent] += 1


def shortest_lengths(data, agent, graph, pprev, prev, row, i, words):
    '''Pass to pathsThruGraphs: compute the shortest length of the navigation
    from A->B
    '''
    debug('shortest_lengths')
    if agent not in data:
        data[agent] = {'sum': 0,'count': 0}
    if prev['target'] in graph and row['target'] in graph:
        data[agent]['sum'] += \
            len(nx.shortest_path(graph, prev['target'], row['target']))
        data[agent]['count'] += 1


def writeResults(log, output):
    debug('writeResults')
    f = open(output, 'w')
    f.write("agent,i,timestamp,rank,length,from,to,class,package\n")
    for line in log:
        # We subtract the 20 seconds that was added earlier to the timestamps
        line['timestamp'] -= datetime.timedelta(seconds=20)
        f.write("%s,%d,%s,%d,%d,%s,%s,%s,%s\n" % \
                    (line['agent'],
                     line['i'],
                     line['timestamp'].time(),
                     line['rank'],
                     line['length'],
                     line['from'],
                     line['to'],
                     line['class'],
                     line['package']))
    f.close()


# If enrich is set to true, it's only valid to run pathsThruGraphs once. You
# must then reset the graph using the loader
enrich = False


def resultsPrev(data, agent, graph, pprev, prev, row, i, words):
    '''Use the previous location to predict the current location.

    >>> nav,g = init()
    >>> r = []
    >>> pathsThruGraphs(nav,g,resultsPrev,data=r)
    >>> writeResults(r,'out.csv')
    '''
    debug('resultsPrev')
    if prev['target'] in graph and row['target'] in graph:
        if pprev and pprev['target'] in graph and enrich:
            graph.add_edge(pprev['target'], prev['target'])
        activation = {prev['target']: 1.0}
        for word in words:
            activation[word]=1.0
        rank, length = getResultRank(graph, activation, row['target'],
                                     iterations=3, d=0.85, navnum=i)
        data.append({'agent': agent,
                     'i': i,
                     'rank': rank,
                     'length': length,
                     'from': prev['target'],
                     'to': row['target'],
                     'class': between_class(prev['target'], row['target']),
                     'package': between_package(prev['target'], row['target']),
                     'timestamp': row['timestamp']})


def resultsNoop(data, agent, graph, pprev, prev, row, i, words):
    '''Use just the words to make predictions.'''
    debug('resultsNoop')
    if prev['target'] in graph and row['target'] in graph:
        activation = {}
        for word in words:
            activation[word] = 1.0
        rank, length = \
            getResultRank(graph, activation, row['target'], navnum=i)
        data.append({'agent': agent,
                     'i': i,
                     'rank': rank,
                     'length': length,
                     'from': prev['target'],
                     'to': row['target'],
                     'class': between_class(prev['target'], row['target']),
                     'package': between_package(prev['target'], row['target']),
                     'timestamp': row['timestamp']})


def resultsPath(data, agent, graph, pprev, prev, row, i, words):
    '''The most recently encountered item has the highest activation, older
    items have lower activation. No accumulation of activation.

    >>> nav,g = init()
    >>> r = {}
    >>> pathsThruGraphs(nav,g,resultsPath,data=r)
    >>> writeResults(r['log'],'out.csv')
    '''
    debug('resultsPath')
    if agent not in data:
        data[agent] = []
    if 'log' not in data:
        data['log'] = []
    if prev['target'] in graph and row['target'] in graph:
        if pprev and pprev['target'] in graph and enrich:
            graph.add_edge(pprev['target'], prev['target'])

        if prev['target'] in data[agent]:
            data[agent].remove(prev['target'])
        data[agent].append(prev['target'])

        a = 1.0
        activation = {}
        for word in words:
            activation[word] = 1.0
        for j in reversed(range(len(data[agent]))):
            activation[data[agent][j]] = a
            a *= 0.9
        rank, length = \
            getResultRank(graph, activation, row['target'], navnum=i)
        data['log'].append({'agent': agent,
                            'i': i,
                            'rank': rank,
                            'length': length,
                            'from': prev['target'],
                            'to': row['target'],
                            'class': between_class(prev['target'],
                                                   row['target']),
                            'package': between_package(prev['target'],
                                                       row['target']),
                            'timestamp': row['timestamp']})


def sorter (x,y):
    debug('sorter')
    return cmp(y[1],x[1])


def wordNode (n):
    debug('wordNode')
    return n[0] == 'word'


def dumpActivation(activation, navnum):
    debug('dumpActivation')
    for (item,val) in activation:
        if wordNode(item):
            activation_root.write("%s|%s|word|%s\n" %( navnum, val, item[1]))
        elif ';.' not in item:
            activation_root.write("%s|%s|class|%s\n" %( navnum, val, item))
        elif '#' in item:
            activation_root.write("%s|%s|localvar|%s\n" %( navnum, val, item))
        else:
            activation_root.write("%s|%s|method|%s\n" %( navnum, val, item))


def getResultRank(graph, start, end, iterations=3, d=0.85, navnum=0):
    debug('getResultRank')
    last = sorted(start.items(),sorter)[0][0]
    activation = spreadingActivation(graph, start, iterations, d)
    dumpActivation(activation, navnum)
    activation = [item for (item,val) in activation if item != '' and \
                      item != last and not wordNode(item) and \
                      '#' not in item and ';.' in item]
    rank = 0
    for item in activation:
        rank += 1
        if item == end:
            break
    return rank, len(activation)


def weight(a, b):
    debug('weight')
    if wordNode(a) or wordNode(b):
        return 0.7
    if normalize(a) == normalize(b):
        return 1
    return 0.7


def spreadingActivation(graph,activation,iterations=3,d=0.85):
    '''Perform spreading activation computation on the graph by activating
    nodes in the activation dictionary. activation = {} where key is node,
    value is initial activation weight
    '''
    debug('spreadingActivation')
    for x in range(iterations):
        for i in activation.keys():
            if i not in graph:
                continue
            w = 1.0 / len(graph.neighbors(i))
            for j in graph.neighbors(i):
                if j not in activation:
                    activation[j] = 0.0
                activation[j] = activation[j] + (activation[i] * w * d)
    return sorted(activation.items(), sorter)


def generateActivation(nav, g, agent):
    debug('generateActivation')
    def ignore(word):
        return word not in ['ibm', 'lcom', 'ljava', 'lang', 'string', 'set',
                            'tostring', 'instanc']

    if len(nav[agent]) == 0:
        return
    prev = nav[agent][0]
    data = []
    for item in nav[agent]:
        if prev in g[agent] and item in g[agent]:
            if prev in data:
                data.remove(prev)
            data.append(prev)
            prev = item
    a = 1.0
    activation = {}
    for i in reversed(range(len(data))):
        activation[data[i]] = a
        a *= 0.9
    return [item[1] for (item, activation) in \
                spreadingActivation(g[agent], activation) \
                if wordNode(item) and ignore(item[1])][0:40]


def bug_report_text():
    debug('bug_report_text')
    BUG_REPORT = '''Hi, when an explicit fold is collapsed go on the line of this fold then delete the line (action delete-line)'''
    wordList = indexCamelWords(BUG_REPORT)
    wordList.extend(indexWords(BUG_REPORT))
    result = []
    for word in wordList:
        result.append(word[1])
    #debug(result)
    return result


def buggy_method_text():
    debug('buggy_method_text')
    BUGGY_METHOD = '''public void updateCaretStatus()
    {
        if (showCaretStatus)
        {
            Buffer buffer = view.getBuffer();

            if(!buffer.isLoaded() ||
                /* can happen when switching buffers sometimes */
                buffer != view.getTextArea().getBuffer())
            {
                caretStatus.setText(" ");
                return;
            }

            JEditTextArea textArea = view.getTextArea();

            int caretPosition = textArea.getCaretPosition();
            int currLine = textArea.getCaretLine();

            // there must be a better way of fixing this...
            // the problem is that this method can sometimes
            // be called as a result of a text area scroll
            // event, in which case the caret position has
            // not been updated yet.
            if(currLine >= buffer.getLineCount())
                return; // hopefully another caret update will come?

            int start = textArea.getLineStartOffset(currLine);
            int dot = caretPosition - start;

            if(dot < 0)
                return;

            int bufferLength = buffer.getLength();

            buffer.getText(start,dot,seg);
            int virtualPosition = StandardUtilities.getVirtualWidth(seg,
                buffer.getTabSize());
            // for GC
            seg.array = null;
            seg.count = 0;

            if (jEdit.getBooleanProperty("view.status.show-caret-linenumber", true))
            {
                buf.append(currLine + 1);
                buf.append(',');
            }
            if (jEdit.getBooleanProperty("view.status.show-caret-dot", true))
            {
                buf.append(dot);
            }
            if (jEdit.getBooleanProperty("view.status.show-caret-virtual", true) &&
                virtualPosition != dot)
            {
                buf.append('-');
                buf.append(virtualPosition + 1);
            }
            if (buf.length() > 0)
            {
                buf.append(' ');
            }
            if (jEdit.getBooleanProperty("view.status.show-caret-offset", true) &&
                jEdit.getBooleanProperty("view.status.show-caret-bufferlength", true))
            {
                buf.append('(');
                buf.append(caretPosition);
                buf.append('/');
                buf.append(bufferLength);
                buf.append(')');
            }
            else if (jEdit.getBooleanProperty("view.status.show-caret-offset", true))
            {
                buf.append('(');
                buf.append(caretPosition);
                buf.append(')');
            }
            else if (jEdit.getBooleanProperty("view.status.show-caret-bufferlength", true))
            {
                buf.append('(');
                buf.append(bufferLength);
                buf.append(')');
            }

            caretStatus.setText(buf.toString());
            buf.setLength(0);
        }
    } //}}}'''
    wordList = indexCamelWords(BUGGY_METHOD)
    wordList.extend(indexWords(BUGGY_METHOD))
    result = []
    for word in wordList:
        result.append(word[1])
    return result


# ID for ParticipantK
k1 = 'a23fe51d-196a-45b3-addf-3db4e8423e4f';
p2 = '4582eb60-df56-4912-872d-03453f296920';
p3 = '897f2b07-9fbe-4e91-add9-b6d2be1c58b4';
p4 = '9b4ad4b4-6216-4e46-92c0-7ea586f1fe1d';
p5 = '0bbc9bc8-2978-4151-a040-507bc560df65';
p6 = 'f81ea274-33c9-4e6c-8aed-55e0cf0e4eac';
p7 = '4ed0c123-dfd8-4400-8813-a9ce888c473d';
p8 = '3ce251fb-7c5e-47eb-948d-9ab121218672';
p9 = 'a37bfdf5-17da-473f-8edd-f4b193ba896d';
p10 = '7a069de8-ca12-4ac1-8803-72b12e640b42';
p11 = '02d4fe4a-661b-4a6c-9196-85fb978616e7';
p12 = '6b4e270c-fe10-40f2-a4aa-a69330dd58cd';


def activate_participantK(g, text):
    debug('activate_participantK')
    words = [node for node in g[p2] if wordNode(node) and node[1] in text]
    return words


# Uncomment the goal text you want to use.
goal_text = bug_report_text()
#goal_text = buggy_method_text()

def participantK_with_history_no_goal():
    debug('Begin participantK_with_history_no_goal')
    global sourcefile, nav, g, activation_root
    sourcefile = 'p02.db'
    activation_root = \
        open('ActivationVectors-WithHistNoGoal.csv',
             'w')
    nav, g = init()
    print 'Analyzing data...'
    r = {}
    pathsThruGraphs(nav, g, resultsPath, data=r)
    writeResults(r['log'],
                 'PfisOutput-WithHistNoGoal.csv')
    activation_root.close()
    debug('End participantK_with_history_no_goal')


def participantK_no_history_no_goal():
    debug('Begin participantK_no_history_no_goal')
    global sourcefile, nav, g, activation_root
    sourcefile = 'p3.db'
    activation_root = \
        open('ActivationVectors-NoHistNoGoal.csv', 'w')
    nav, g = init()
    print 'Analyzing data...'
    r = []
    pathsThruGraphs(nav, g, resultsPrev, data=r)
    writeResults(r, 'PfisOutput-NoHistNoGoal.csv')
    activation_root.close()
    debug('End participantK_no_history_no_goal')

def participantK_with_history_with_goal():
    debug('Begin participantK_with_history_with_goal')
    global sourcefile, nav, g, activation_root
    sourcefile = 'p3.db'
    activation_root = \
        open('ActivationVectors-WithHistWithGoal.csv',
             'w')
    nav, g = init()
    print 'Analyzing data...'
    r = {}
    pathsThruGraphs(nav, g, resultsPath, data=r,
                    words=activate_participantK(g, goal_text))
    writeResults(r['log'],
                 'PfisOutput-WithHistWithGoal.csv')
    activation_root.close()
    debug('End participantK_with_history_with_goal')

def participantK_no_history_with_goal():
    debug('Begin participantK_no_history_with_goal')
    global sourcefile, nav, g, activation_root
    sourcefile = 'p3.db'
    activation_root = \
        open('ActivationVectors-NoHistWithGoal.csv',
             'w')
    nav, g = init()
    print 'Analyzing data...'
    r = []
    pathsThruGraphs(nav, g, resultsPrev, data=r,
                    words=activate_participantK(g, goal_text))
    writeResults(r, 'PfisOutput-NoHistWithGoal.csv')
    activation_root.close()
    debug('End participantK_no_history_with_goal')

print '----------------------------------------'
print 'Processing H- (with history, no goal)'
participantK_with_history_no_goal()
print '----------------------------------------'
#print 'Processing HG (with history, with goal)'
#participantK_with_history_with_goal()
#print '----------------------------------------'
#print 'Processing -G (no history, with goal)'
#participantK_no_history_with_goal()
#print '----------------------------------------'
#print 'Processing -- (no history, no goal)'
#participantK_no_history_no_goal()
#print '----------------------------------------'


sys.exit(0)
