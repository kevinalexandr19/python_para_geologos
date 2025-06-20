(function() {
var __webpack_modules__ = {
"800": (function (__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {
"use strict";
__webpack_require__.r(__webpack_exports__);
__webpack_require__.d(__webpack_exports__, {
  v4: function() { return v4; }
});
var IDX = 256, HEX = [], BUFFER;
while(IDX--)HEX[IDX] = (IDX + 256).toString(16).substring(1);
function v4() {
    var i = 0, num, out = '';
    if (!BUFFER || IDX + 16 > 256) {
        BUFFER = Array(i = 256);
        while(i--)BUFFER[i] = 256 * Math.random() | 0;
        i = IDX = 0;
    }
    for(; i < 16; i++){
        num = BUFFER[IDX + i];
        if (i == 6) out += HEX[num & 15 | 64];
        else if (i == 8) out += HEX[num & 63 | 128];
        else out += HEX[num];
        if (i & 1 && i > 1 && i < 11) out += '-';
    }
    IDX++;
    return out;
}
}),
"336": (function (__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {
"use strict";
__webpack_require__.r(__webpack_exports__);
__webpack_require__.d(__webpack_exports__, {
  createEffect: function() { return createEffect; },
  createResource: function() { return createResource; },
  createRoot: function() { return createRoot; },
  createSignal: function() { return createSignal; }
});
let taskIdCounter = 1, isCallbackScheduled = false, isPerformingWork = false, taskQueue = [], currentTask = null, shouldYieldToHost = null, yieldInterval = 5, deadline = 0, maxYieldInterval = 300, scheduleCallback = null, scheduledCallback = null;
const maxSigned31BitInt = 1073741823;
function setupScheduler() {
    const channel = new MessageChannel(), port = channel.port2;
    scheduleCallback = ()=>port.postMessage(null);
    channel.port1.onmessage = ()=>{
        if (scheduledCallback !== null) {
            const currentTime = performance.now();
            deadline = currentTime + yieldInterval;
            const hasTimeRemaining = true;
            try {
                const hasMoreWork = scheduledCallback(hasTimeRemaining, currentTime);
                if (!hasMoreWork) scheduledCallback = null;
                else port.postMessage(null);
            } catch (error) {
                port.postMessage(null);
                throw error;
            }
        }
    };
    if (navigator && navigator.scheduling && navigator.scheduling.isInputPending) {
        const scheduling = navigator.scheduling;
        shouldYieldToHost = ()=>{
            const currentTime = performance.now();
            if (currentTime >= deadline) {
                if (scheduling.isInputPending()) return true;
                return currentTime >= maxYieldInterval;
            } else return false;
        };
    } else shouldYieldToHost = ()=>performance.now() >= deadline;
}
function enqueue(taskQueue, task) {
    function findIndex() {
        let m = 0;
        let n = taskQueue.length - 1;
        while(m <= n){
            const k = n + m >> 1;
            const cmp = task.expirationTime - taskQueue[k].expirationTime;
            if (cmp > 0) m = k + 1;
            else if (cmp < 0) n = k - 1;
            else return k;
        }
        return m;
    }
    taskQueue.splice(findIndex(), 0, task);
}
function requestCallback(fn, options) {
    if (!scheduleCallback) setupScheduler();
    let startTime = performance.now(), timeout = maxSigned31BitInt;
    if (options && options.timeout) timeout = options.timeout;
    const newTask = {
        id: taskIdCounter++,
        fn,
        startTime,
        expirationTime: startTime + timeout
    };
    enqueue(taskQueue, newTask);
    if (!isCallbackScheduled && !isPerformingWork) {
        isCallbackScheduled = true;
        scheduledCallback = flushWork;
        scheduleCallback();
    }
    return newTask;
}
function cancelCallback(task) {
    task.fn = null;
}
function flushWork(hasTimeRemaining, initialTime) {
    isCallbackScheduled = false;
    isPerformingWork = true;
    try {
        return workLoop(hasTimeRemaining, initialTime);
    } finally{
        currentTask = null;
        isPerformingWork = false;
    }
}
function workLoop(hasTimeRemaining, initialTime) {
    let currentTime = initialTime;
    currentTask = taskQueue[0] || null;
    while(currentTask !== null){
        if (currentTask.expirationTime > currentTime && (!hasTimeRemaining || shouldYieldToHost())) break;
        const callback = currentTask.fn;
        if (callback !== null) {
            currentTask.fn = null;
            const didUserCallbackTimeout = currentTask.expirationTime <= currentTime;
            callback(didUserCallbackTimeout);
            currentTime = performance.now();
            if (currentTask === taskQueue[0]) taskQueue.shift();
        } else taskQueue.shift();
        currentTask = taskQueue[0] || null;
    }
    return currentTask !== null;
}
const sharedConfig = {
    context: undefined,
    registry: undefined
};
function setHydrateContext(context) {
    sharedConfig.context = context;
}
function nextHydrateContext() {
    return {
        ...sharedConfig.context,
        id: `${sharedConfig.context.id}${sharedConfig.context.count++}-`,
        count: 0
    };
}
const equalFn = (a, b)=>a === b;
const $PROXY = Symbol("solid-proxy");
const $TRACK = Symbol("solid-track");
const $DEVCOMP = Symbol("solid-dev-component");
const signalOptions = {
    equals: equalFn
};
let ERROR = null;
let runEffects = runQueue;
const STALE = 1;
const PENDING = 2;
const UNOWNED = {
    owned: null,
    cleanups: null,
    context: null,
    owner: null
};
const NO_INIT = {};
var Owner = null;
let Transition = null;
let Scheduler = null;
let ExternalSourceConfig = null;
let Listener = null;
let Updates = null;
let Effects = null;
let ExecCount = 0;
function createRoot(fn, detachedOwner) {
    const listener = Listener, owner = Owner, unowned = fn.length === 0, current = detachedOwner === undefined ? owner : detachedOwner, root = unowned ? UNOWNED : {
        owned: null,
        cleanups: null,
        context: current ? current.context : null,
        owner: current
    }, updateFn = unowned ? fn : ()=>fn(()=>untrack(()=>cleanNode(root)));
    Owner = root;
    Listener = null;
    try {
        return runUpdates(updateFn, true);
    } finally{
        Listener = listener;
        Owner = owner;
    }
}
function createSignal(value, options) {
    options = options ? Object.assign({}, signalOptions, options) : signalOptions;
    const s = {
        value,
        observers: null,
        observerSlots: null,
        comparator: options.equals || undefined
    };
    const setter = (value)=>{
        if (typeof value === "function") {
            if (Transition && Transition.running && Transition.sources.has(s)) value = value(s.tValue);
            else value = value(s.value);
        }
        return writeSignal(s, value);
    };
    return [
        readSignal.bind(s),
        setter
    ];
}
function createComputed(fn, value, options) {
    const c = createComputation(fn, value, true, STALE);
    if (Scheduler && Transition && Transition.running) Updates.push(c);
    else updateComputation(c);
}
function createRenderEffect(fn, value, options) {
    const c = createComputation(fn, value, false, STALE);
    if (Scheduler && Transition && Transition.running) Updates.push(c);
    else updateComputation(c);
}
function createEffect(fn, value, options) {
    runEffects = runUserEffects;
    const c = createComputation(fn, value, false, STALE), s = SuspenseContext && useContext(SuspenseContext);
    if (s) c.suspense = s;
    if (!options || !options.render) c.user = true;
    Effects ? Effects.push(c) : updateComputation(c);
}
function createReaction(onInvalidate, options) {
    let fn;
    const c = createComputation(()=>{
        fn ? fn() : untrack(onInvalidate);
        fn = undefined;
    }, undefined, false, 0), s = SuspenseContext && useContext(SuspenseContext);
    if (s) c.suspense = s;
    c.user = true;
    return (tracking)=>{
        fn = tracking;
        updateComputation(c);
    };
}
function createMemo(fn, value, options) {
    options = options ? Object.assign({}, signalOptions, options) : signalOptions;
    const c = createComputation(fn, value, true, 0);
    c.observers = null;
    c.observerSlots = null;
    c.comparator = options.equals || undefined;
    if (Scheduler && Transition && Transition.running) {
        c.tState = STALE;
        Updates.push(c);
    } else updateComputation(c);
    return readSignal.bind(c);
}
function isPromise(v) {
    return v && typeof v === "object" && "then" in v;
}
function createResource(pSource, pFetcher, pOptions) {
    let source;
    let fetcher;
    let options;
    if (arguments.length === 2 && typeof pFetcher === "object" || arguments.length === 1) {
        source = true;
        fetcher = pSource;
        options = pFetcher || {};
    } else {
        source = pSource;
        fetcher = pFetcher;
        options = pOptions || {};
    }
    let pr = null, initP = NO_INIT, id = null, loadedUnderTransition = false, scheduled = false, resolved = "initialValue" in options, dynamic = typeof source === "function" && createMemo(source);
    const contexts = new Set(), [value, setValue] = (options.storage || createSignal)(options.initialValue), [error, setError] = createSignal(undefined), [track, trigger] = createSignal(undefined, {
        equals: false
    }), [state, setState] = createSignal(resolved ? "ready" : "unresolved");
    if (sharedConfig.context) {
        id = `${sharedConfig.context.id}${sharedConfig.context.count++}`;
        let v;
        if (options.ssrLoadFrom === "initial") initP = options.initialValue;
        else if (sharedConfig.load && (v = sharedConfig.load(id))) initP = v;
    }
    function loadEnd(p, v, error, key) {
        if (pr === p) {
            pr = null;
            key !== undefined && (resolved = true);
            if ((p === initP || v === initP) && options.onHydrated) queueMicrotask(()=>options.onHydrated(key, {
                    value: v
                }));
            initP = NO_INIT;
            if (Transition && p && loadedUnderTransition) {
                Transition.promises.delete(p);
                loadedUnderTransition = false;
                runUpdates(()=>{
                    Transition.running = true;
                    completeLoad(v, error);
                }, false);
            } else completeLoad(v, error);
        }
        return v;
    }
    function completeLoad(v, err) {
        runUpdates(()=>{
            if (err === undefined) setValue(()=>v);
            setState(err !== undefined ? "errored" : resolved ? "ready" : "unresolved");
            setError(err);
            for (const c of contexts.keys())c.decrement();
            contexts.clear();
        }, false);
    }
    function read() {
        const c = SuspenseContext && useContext(SuspenseContext), v = value(), err = error();
        if (err !== undefined && !pr) throw err;
        if (Listener && !Listener.user && c) createComputed(()=>{
            track();
            if (pr) {
                if (c.resolved && Transition && loadedUnderTransition) Transition.promises.add(pr);
                else if (!contexts.has(c)) {
                    c.increment();
                    contexts.add(c);
                }
            }
        });
        return v;
    }
    function load(refetching = true) {
        if (refetching !== false && scheduled) return;
        scheduled = false;
        const lookup = dynamic ? dynamic() : source;
        loadedUnderTransition = Transition && Transition.running;
        if (lookup == null || lookup === false) {
            loadEnd(pr, untrack(value));
            return;
        }
        if (Transition && pr) Transition.promises.delete(pr);
        const p = initP !== NO_INIT ? initP : untrack(()=>fetcher(lookup, {
                value: value(),
                refetching
            }));
        if (!isPromise(p)) {
            loadEnd(pr, p, undefined, lookup);
            return p;
        }
        pr = p;
        if ("value" in p) {
            if (p.status === "success") loadEnd(pr, p.value, undefined, lookup);
            else loadEnd(pr, undefined, undefined, lookup);
            return p;
        }
        scheduled = true;
        queueMicrotask(()=>scheduled = false);
        runUpdates(()=>{
            setState(resolved ? "refreshing" : "pending");
            trigger();
        }, false);
        return p.then((v)=>loadEnd(p, v, undefined, lookup), (e)=>loadEnd(p, undefined, castError(e), lookup));
    }
    Object.defineProperties(read, {
        state: {
            get: ()=>state()
        },
        error: {
            get: ()=>error()
        },
        loading: {
            get () {
                const s = state();
                return s === "pending" || s === "refreshing";
            }
        },
        latest: {
            get () {
                if (!resolved) return read();
                const err = error();
                if (err && !pr) throw err;
                return value();
            }
        }
    });
    if (dynamic) createComputed(()=>load(false));
    else load(false);
    return [
        read,
        {
            refetch: load,
            mutate: setValue
        }
    ];
}
function createDeferred(source, options) {
    let t, timeout = options ? options.timeoutMs : undefined;
    const node = createComputation(()=>{
        if (!t || !t.fn) t = requestCallback(()=>setDeferred(()=>node.value), timeout !== undefined ? {
            timeout
        } : undefined);
        return source();
    }, undefined, true);
    const [deferred, setDeferred] = createSignal(Transition && Transition.running && Transition.sources.has(node) ? node.tValue : node.value, options);
    updateComputation(node);
    setDeferred(()=>Transition && Transition.running && Transition.sources.has(node) ? node.tValue : node.value);
    return deferred;
}
function createSelector(source, fn = equalFn, options) {
    const subs = new Map();
    const node = createComputation((p)=>{
        const v = source();
        for (const [key, val] of subs.entries())if (fn(key, v) !== fn(key, p)) for (const c of val.values()){
            c.state = STALE;
            if (c.pure) Updates.push(c);
            else Effects.push(c);
        }
        return v;
    }, undefined, true, STALE);
    updateComputation(node);
    return (key)=>{
        const listener = Listener;
        if (listener) {
            let l;
            if (l = subs.get(key)) l.add(listener);
            else subs.set(key, l = new Set([
                listener
            ]));
            onCleanup(()=>{
                l.delete(listener);
                !l.size && subs.delete(key);
            });
        }
        return fn(key, Transition && Transition.running && Transition.sources.has(node) ? node.tValue : node.value);
    };
}
function batch(fn) {
    return runUpdates(fn, false);
}
function untrack(fn) {
    if (!ExternalSourceConfig && Listener === null) return fn();
    const listener = Listener;
    Listener = null;
    try {
        if (ExternalSourceConfig) return ExternalSourceConfig.untrack(fn);
        return fn();
    } finally{
        Listener = listener;
    }
}
function on(deps, fn, options) {
    const isArray = Array.isArray(deps);
    let prevInput;
    let defer = options && options.defer;
    return (prevValue)=>{
        let input;
        if (isArray) {
            input = Array(deps.length);
            for(let i = 0; i < deps.length; i++)input[i] = deps[i]();
        } else input = deps();
        if (defer) {
            defer = false;
            return prevValue;
        }
        const result = untrack(()=>fn(input, prevInput, prevValue));
        prevInput = input;
        return result;
    };
}
function onMount(fn) {
    createEffect(()=>untrack(fn));
}
function onCleanup(fn) {
    if (Owner === null) ;
    else if (Owner.cleanups === null) Owner.cleanups = [
        fn
    ];
    else Owner.cleanups.push(fn);
    return fn;
}
function catchError(fn, handler) {
    ERROR || (ERROR = Symbol("error"));
    Owner = createComputation(undefined, undefined, true);
    Owner.context = {
        ...Owner.context,
        [ERROR]: [
            handler
        ]
    };
    if (Transition && Transition.running) Transition.sources.add(Owner);
    try {
        return fn();
    } catch (err) {
        handleError(err);
    } finally{
        Owner = Owner.owner;
    }
}
function getListener() {
    return Listener;
}
function getOwner() {
    return Owner;
}
function runWithOwner(o, fn) {
    const prev = Owner;
    const prevListener = Listener;
    Owner = o;
    Listener = null;
    try {
        return runUpdates(fn, true);
    } catch (err) {
        handleError(err);
    } finally{
        Owner = prev;
        Listener = prevListener;
    }
}
function enableScheduling(scheduler = requestCallback) {
    Scheduler = scheduler;
}
function startTransition(fn) {
    if (Transition && Transition.running) {
        fn();
        return Transition.done;
    }
    const l = Listener;
    const o = Owner;
    return Promise.resolve().then(()=>{
        Listener = l;
        Owner = o;
        let t;
        if (Scheduler || SuspenseContext) {
            t = Transition || (Transition = {
                sources: new Set(),
                effects: [],
                promises: new Set(),
                disposed: new Set(),
                queue: new Set(),
                running: true
            });
            t.done || (t.done = new Promise((res)=>t.resolve = res));
            t.running = true;
        }
        runUpdates(fn, false);
        Listener = Owner = null;
        return t ? t.done : undefined;
    });
}
const [transPending, setTransPending] = /*@__PURE__*/ createSignal(false);
function useTransition() {
    return [
        transPending,
        startTransition
    ];
}
function resumeEffects(e) {
    Effects.push.apply(Effects, e);
    e.length = 0;
}
function createContext(defaultValue, options) {
    const id = Symbol("context");
    return {
        id,
        Provider: createProvider(id),
        defaultValue
    };
}
function useContext(context) {
    return Owner && Owner.context && Owner.context[context.id] !== undefined ? Owner.context[context.id] : context.defaultValue;
}
function children(fn) {
    const children = createMemo(fn);
    const memo = createMemo(()=>resolveChildren(children()));
    memo.toArray = ()=>{
        const c = memo();
        return Array.isArray(c) ? c : c != null ? [
            c
        ] : [];
    };
    return memo;
}
let SuspenseContext;
function getSuspenseContext() {
    return SuspenseContext || (SuspenseContext = createContext());
}
function enableExternalSource(factory, untrack = (fn)=>fn()) {
    if (ExternalSourceConfig) {
        const { factory: oldFactory, untrack: oldUntrack } = ExternalSourceConfig;
        ExternalSourceConfig = {
            factory: (fn, trigger)=>{
                const oldSource = oldFactory(fn, trigger);
                const source = factory((x)=>oldSource.track(x), trigger);
                return {
                    track: (x)=>source.track(x),
                    dispose () {
                        source.dispose();
                        oldSource.dispose();
                    }
                };
            },
            untrack: (fn)=>oldUntrack(()=>untrack(fn))
        };
    } else ExternalSourceConfig = {
        factory,
        untrack
    };
}
function readSignal() {
    const runningTransition = Transition && Transition.running;
    if (this.sources && (runningTransition ? this.tState : this.state)) {
        if ((runningTransition ? this.tState : this.state) === STALE) updateComputation(this);
        else {
            const updates = Updates;
            Updates = null;
            runUpdates(()=>lookUpstream(this), false);
            Updates = updates;
        }
    }
    if (Listener) {
        const sSlot = this.observers ? this.observers.length : 0;
        if (!Listener.sources) {
            Listener.sources = [
                this
            ];
            Listener.sourceSlots = [
                sSlot
            ];
        } else {
            Listener.sources.push(this);
            Listener.sourceSlots.push(sSlot);
        }
        if (!this.observers) {
            this.observers = [
                Listener
            ];
            this.observerSlots = [
                Listener.sources.length - 1
            ];
        } else {
            this.observers.push(Listener);
            this.observerSlots.push(Listener.sources.length - 1);
        }
    }
    if (runningTransition && Transition.sources.has(this)) return this.tValue;
    return this.value;
}
function writeSignal(node, value, isComp) {
    let current = Transition && Transition.running && Transition.sources.has(node) ? node.tValue : node.value;
    if (!node.comparator || !node.comparator(current, value)) {
        if (Transition) {
            const TransitionRunning = Transition.running;
            if (TransitionRunning || !isComp && Transition.sources.has(node)) {
                Transition.sources.add(node);
                node.tValue = value;
            }
            if (!TransitionRunning) node.value = value;
        } else node.value = value;
        if (node.observers && node.observers.length) runUpdates(()=>{
            for(let i = 0; i < node.observers.length; i += 1){
                const o = node.observers[i];
                const TransitionRunning = Transition && Transition.running;
                if (TransitionRunning && Transition.disposed.has(o)) continue;
                if (TransitionRunning ? !o.tState : !o.state) {
                    if (o.pure) Updates.push(o);
                    else Effects.push(o);
                    if (o.observers) markDownstream(o);
                }
                if (!TransitionRunning) o.state = STALE;
                else o.tState = STALE;
            }
            if (Updates.length > 10e5) {
                Updates = [];
                throw new Error();
            }
        }, false);
    }
    return value;
}
function updateComputation(node) {
    if (!node.fn) return;
    cleanNode(node);
    const time = ExecCount;
    runComputation(node, Transition && Transition.running && Transition.sources.has(node) ? node.tValue : node.value, time);
    if (Transition && !Transition.running && Transition.sources.has(node)) queueMicrotask(()=>{
        runUpdates(()=>{
            Transition && (Transition.running = true);
            Listener = Owner = node;
            runComputation(node, node.tValue, time);
            Listener = Owner = null;
        }, false);
    });
}
function runComputation(node, value, time) {
    let nextValue;
    const owner = Owner, listener = Listener;
    Listener = Owner = node;
    try {
        nextValue = node.fn(value);
    } catch (err) {
        if (node.pure) {
            if (Transition && Transition.running) {
                node.tState = STALE;
                node.tOwned && node.tOwned.forEach(cleanNode);
                node.tOwned = undefined;
            } else {
                node.state = STALE;
                node.owned && node.owned.forEach(cleanNode);
                node.owned = null;
            }
        }
        node.updatedAt = time + 1;
        return handleError(err);
    } finally{
        Listener = listener;
        Owner = owner;
    }
    if (!node.updatedAt || node.updatedAt <= time) {
        if (node.updatedAt != null && "observers" in node) writeSignal(node, nextValue, true);
        else if (Transition && Transition.running && node.pure) {
            Transition.sources.add(node);
            node.tValue = nextValue;
        } else node.value = nextValue;
        node.updatedAt = time;
    }
}
function createComputation(fn, init, pure, state = STALE, options) {
    const c = {
        fn,
        state: state,
        updatedAt: null,
        owned: null,
        sources: null,
        sourceSlots: null,
        cleanups: null,
        value: init,
        owner: Owner,
        context: Owner ? Owner.context : null,
        pure
    };
    if (Transition && Transition.running) {
        c.state = 0;
        c.tState = state;
    }
    if (Owner === null) ;
    else if (Owner !== UNOWNED) {
        if (Transition && Transition.running && Owner.pure) {
            if (!Owner.tOwned) Owner.tOwned = [
                c
            ];
            else Owner.tOwned.push(c);
        } else if (!Owner.owned) Owner.owned = [
            c
        ];
        else Owner.owned.push(c);
    }
    if (ExternalSourceConfig && c.fn) {
        const [track, trigger] = createSignal(undefined, {
            equals: false
        });
        const ordinary = ExternalSourceConfig.factory(c.fn, trigger);
        onCleanup(()=>ordinary.dispose());
        const triggerInTransition = ()=>startTransition(trigger).then(()=>inTransition.dispose());
        const inTransition = ExternalSourceConfig.factory(c.fn, triggerInTransition);
        c.fn = (x)=>{
            track();
            return Transition && Transition.running ? inTransition.track(x) : ordinary.track(x);
        };
    }
    return c;
}
function runTop(node) {
    const runningTransition = Transition && Transition.running;
    if ((runningTransition ? node.tState : node.state) === 0) return;
    if ((runningTransition ? node.tState : node.state) === PENDING) return lookUpstream(node);
    if (node.suspense && untrack(node.suspense.inFallback)) return node.suspense.effects.push(node);
    const ancestors = [
        node
    ];
    while((node = node.owner) && (!node.updatedAt || node.updatedAt < ExecCount)){
        if (runningTransition && Transition.disposed.has(node)) return;
        if (runningTransition ? node.tState : node.state) ancestors.push(node);
    }
    for(let i = ancestors.length - 1; i >= 0; i--){
        node = ancestors[i];
        if (runningTransition) {
            let top = node, prev = ancestors[i + 1];
            while((top = top.owner) && top !== prev){
                if (Transition.disposed.has(top)) return;
            }
        }
        if ((runningTransition ? node.tState : node.state) === STALE) updateComputation(node);
        else if ((runningTransition ? node.tState : node.state) === PENDING) {
            const updates = Updates;
            Updates = null;
            runUpdates(()=>lookUpstream(node, ancestors[0]), false);
            Updates = updates;
        }
    }
}
function runUpdates(fn, init) {
    if (Updates) return fn();
    let wait = false;
    if (!init) Updates = [];
    if (Effects) wait = true;
    else Effects = [];
    ExecCount++;
    try {
        const res = fn();
        completeUpdates(wait);
        return res;
    } catch (err) {
        if (!wait) Effects = null;
        Updates = null;
        handleError(err);
    }
}
function completeUpdates(wait) {
    if (Updates) {
        if (Scheduler && Transition && Transition.running) scheduleQueue(Updates);
        else runQueue(Updates);
        Updates = null;
    }
    if (wait) return;
    let res;
    if (Transition) {
        if (!Transition.promises.size && !Transition.queue.size) {
            const sources = Transition.sources;
            const disposed = Transition.disposed;
            Effects.push.apply(Effects, Transition.effects);
            res = Transition.resolve;
            for (const e of Effects){
                "tState" in e && (e.state = e.tState);
                delete e.tState;
            }
            Transition = null;
            runUpdates(()=>{
                for (const d of disposed)cleanNode(d);
                for (const v of sources){
                    v.value = v.tValue;
                    if (v.owned) for(let i = 0, len = v.owned.length; i < len; i++)cleanNode(v.owned[i]);
                    if (v.tOwned) v.owned = v.tOwned;
                    delete v.tValue;
                    delete v.tOwned;
                    v.tState = 0;
                }
                setTransPending(false);
            }, false);
        } else if (Transition.running) {
            Transition.running = false;
            Transition.effects.push.apply(Transition.effects, Effects);
            Effects = null;
            setTransPending(true);
            return;
        }
    }
    const e = Effects;
    Effects = null;
    if (e.length) runUpdates(()=>runEffects(e), false);
    if (res) res();
}
function runQueue(queue) {
    for(let i = 0; i < queue.length; i++)runTop(queue[i]);
}
function scheduleQueue(queue) {
    for(let i = 0; i < queue.length; i++){
        const item = queue[i];
        const tasks = Transition.queue;
        if (!tasks.has(item)) {
            tasks.add(item);
            Scheduler(()=>{
                tasks.delete(item);
                runUpdates(()=>{
                    Transition.running = true;
                    runTop(item);
                }, false);
                Transition && (Transition.running = false);
            });
        }
    }
}
function runUserEffects(queue) {
    let i, userLength = 0;
    for(i = 0; i < queue.length; i++){
        const e = queue[i];
        if (!e.user) runTop(e);
        else queue[userLength++] = e;
    }
    if (sharedConfig.context) {
        if (sharedConfig.count) {
            sharedConfig.effects || (sharedConfig.effects = []);
            sharedConfig.effects.push(...queue.slice(0, userLength));
            return;
        } else if (sharedConfig.effects) {
            queue = [
                ...sharedConfig.effects,
                ...queue
            ];
            userLength += sharedConfig.effects.length;
            delete sharedConfig.effects;
        }
        setHydrateContext();
    }
    for(i = 0; i < userLength; i++)runTop(queue[i]);
}
function lookUpstream(node, ignore) {
    const runningTransition = Transition && Transition.running;
    if (runningTransition) node.tState = 0;
    else node.state = 0;
    for(let i = 0; i < node.sources.length; i += 1){
        const source = node.sources[i];
        if (source.sources) {
            const state = runningTransition ? source.tState : source.state;
            if (state === STALE) {
                if (source !== ignore && (!source.updatedAt || source.updatedAt < ExecCount)) runTop(source);
            } else if (state === PENDING) lookUpstream(source, ignore);
        }
    }
}
function markDownstream(node) {
    const runningTransition = Transition && Transition.running;
    for(let i = 0; i < node.observers.length; i += 1){
        const o = node.observers[i];
        if (runningTransition ? !o.tState : !o.state) {
            if (runningTransition) o.tState = PENDING;
            else o.state = PENDING;
            if (o.pure) Updates.push(o);
            else Effects.push(o);
            o.observers && markDownstream(o);
        }
    }
}
function cleanNode(node) {
    let i;
    if (node.sources) while(node.sources.length){
        const source = node.sources.pop(), index = node.sourceSlots.pop(), obs = source.observers;
        if (obs && obs.length) {
            const n = obs.pop(), s = source.observerSlots.pop();
            if (index < obs.length) {
                n.sourceSlots[s] = index;
                obs[index] = n;
                source.observerSlots[index] = s;
            }
        }
    }
    if (Transition && Transition.running && node.pure) {
        if (node.tOwned) {
            for(i = node.tOwned.length - 1; i >= 0; i--)cleanNode(node.tOwned[i]);
            delete node.tOwned;
        }
        reset(node, true);
    } else if (node.owned) {
        for(i = node.owned.length - 1; i >= 0; i--)cleanNode(node.owned[i]);
        node.owned = null;
    }
    if (node.cleanups) {
        for(i = node.cleanups.length - 1; i >= 0; i--)node.cleanups[i]();
        node.cleanups = null;
    }
    if (Transition && Transition.running) node.tState = 0;
    else node.state = 0;
}
function reset(node, top) {
    if (!top) {
        node.tState = 0;
        Transition.disposed.add(node);
    }
    if (node.owned) for(let i = 0; i < node.owned.length; i++)reset(node.owned[i]);
}
function castError(err) {
    if (err instanceof Error) return err;
    return new Error(typeof err === "string" ? err : "Unknown error", {
        cause: err
    });
}
function runErrors(err, fns, owner) {
    try {
        for (const f of fns)f(err);
    } catch (e) {
        handleError(e, owner && owner.owner || null);
    }
}
function handleError(err, owner = Owner) {
    const fns = ERROR && owner && owner.context && owner.context[ERROR];
    const error = castError(err);
    if (!fns) throw error;
    if (Effects) Effects.push({
        fn () {
            runErrors(error, fns, owner);
        },
        state: STALE
    });
    else runErrors(error, fns, owner);
}
function resolveChildren(children) {
    if (typeof children === "function" && !children.length) return resolveChildren(children());
    if (Array.isArray(children)) {
        const results = [];
        for(let i = 0; i < children.length; i++){
            const result = resolveChildren(children[i]);
            Array.isArray(result) ? results.push.apply(results, result) : results.push(result);
        }
        return results;
    }
    return children;
}
function createProvider(id, options) {
    return function provider(props) {
        let res;
        createRenderEffect(()=>res = untrack(()=>{
                Owner.context = {
                    ...Owner.context,
                    [id]: props.value
                };
                return children(()=>props.children);
            }), undefined);
        return res;
    };
}
function onError(fn) {
    ERROR || (ERROR = Symbol("error"));
    if (Owner === null) ;
    else if (Owner.context === null || !Owner.context[ERROR]) {
        Owner.context = {
            ...Owner.context,
            [ERROR]: [
                fn
            ]
        };
        mutateContext(Owner, ERROR, [
            fn
        ]);
    } else Owner.context[ERROR].push(fn);
}
function mutateContext(o, key, value) {
    if (o.owned) for(let i = 0; i < o.owned.length; i++){
        if (o.owned[i].context === o.context) mutateContext(o.owned[i], key, value);
        if (!o.owned[i].context) {
            o.owned[i].context = o.context;
            mutateContext(o.owned[i], key, value);
        } else if (!o.owned[i].context[key]) {
            o.owned[i].context[key] = value;
            mutateContext(o.owned[i], key, value);
        }
    }
}
function observable(input) {
    return {
        subscribe (observer) {
            if (!(observer instanceof Object) || observer == null) throw new TypeError("Expected the observer to be an object.");
            const handler = typeof observer === "function" ? observer : observer.next && observer.next.bind(observer);
            if (!handler) return {
                unsubscribe () {}
            };
            const dispose = createRoot((disposer)=>{
                createEffect(()=>{
                    const v = input();
                    untrack(()=>handler(v));
                });
                return disposer;
            });
            if (getOwner()) onCleanup(dispose);
            return {
                unsubscribe () {
                    dispose();
                }
            };
        },
        [Symbol.observable || "@@observable"] () {
            return this;
        }
    };
}
function from(producer) {
    const [s, set] = createSignal(undefined, {
        equals: false
    });
    if ("subscribe" in producer) {
        const unsub = producer.subscribe((v)=>set(()=>v));
        onCleanup(()=>"unsubscribe" in unsub ? unsub.unsubscribe() : unsub());
    } else {
        const clean = producer(set);
        onCleanup(clean);
    }
    return s;
}
const FALLBACK = Symbol("fallback");
function dispose(d) {
    for(let i = 0; i < d.length; i++)d[i]();
}
function mapArray(list, mapFn, options = {}) {
    let items = [], mapped = [], disposers = [], len = 0, indexes = mapFn.length > 1 ? [] : null;
    onCleanup(()=>dispose(disposers));
    return ()=>{
        let newItems = list() || [], i, j;
        newItems[$TRACK];
        return untrack(()=>{
            let newLen = newItems.length, newIndices, newIndicesNext, temp, tempdisposers, tempIndexes, start, end, newEnd, item;
            if (newLen === 0) {
                if (len !== 0) {
                    dispose(disposers);
                    disposers = [];
                    items = [];
                    mapped = [];
                    len = 0;
                    indexes && (indexes = []);
                }
                if (options.fallback) {
                    items = [
                        FALLBACK
                    ];
                    mapped[0] = createRoot((disposer)=>{
                        disposers[0] = disposer;
                        return options.fallback();
                    });
                    len = 1;
                }
            } else if (len === 0) {
                mapped = new Array(newLen);
                for(j = 0; j < newLen; j++){
                    items[j] = newItems[j];
                    mapped[j] = createRoot(mapper);
                }
                len = newLen;
            } else {
                temp = new Array(newLen);
                tempdisposers = new Array(newLen);
                indexes && (tempIndexes = new Array(newLen));
                for(start = 0, end = Math.min(len, newLen); start < end && items[start] === newItems[start]; start++);
                for(end = len - 1, newEnd = newLen - 1; end >= start && newEnd >= start && items[end] === newItems[newEnd]; end--, newEnd--){
                    temp[newEnd] = mapped[end];
                    tempdisposers[newEnd] = disposers[end];
                    indexes && (tempIndexes[newEnd] = indexes[end]);
                }
                newIndices = new Map();
                newIndicesNext = new Array(newEnd + 1);
                for(j = newEnd; j >= start; j--){
                    item = newItems[j];
                    i = newIndices.get(item);
                    newIndicesNext[j] = i === undefined ? -1 : i;
                    newIndices.set(item, j);
                }
                for(i = start; i <= end; i++){
                    item = items[i];
                    j = newIndices.get(item);
                    if (j !== undefined && j !== -1) {
                        temp[j] = mapped[i];
                        tempdisposers[j] = disposers[i];
                        indexes && (tempIndexes[j] = indexes[i]);
                        j = newIndicesNext[j];
                        newIndices.set(item, j);
                    } else disposers[i]();
                }
                for(j = start; j < newLen; j++)if (j in temp) {
                    mapped[j] = temp[j];
                    disposers[j] = tempdisposers[j];
                    if (indexes) {
                        indexes[j] = tempIndexes[j];
                        indexes[j](j);
                    }
                } else mapped[j] = createRoot(mapper);
                mapped = mapped.slice(0, len = newLen);
                items = newItems.slice(0);
            }
            return mapped;
        });
        function mapper(disposer) {
            disposers[j] = disposer;
            if (indexes) {
                const [s, set] = createSignal(j);
                indexes[j] = set;
                return mapFn(newItems[j], s);
            }
            return mapFn(newItems[j]);
        }
    };
}
function indexArray(list, mapFn, options = {}) {
    let items = [], mapped = [], disposers = [], signals = [], len = 0, i;
    onCleanup(()=>dispose(disposers));
    return ()=>{
        const newItems = list() || [];
        newItems[$TRACK];
        return untrack(()=>{
            if (newItems.length === 0) {
                if (len !== 0) {
                    dispose(disposers);
                    disposers = [];
                    items = [];
                    mapped = [];
                    len = 0;
                    signals = [];
                }
                if (options.fallback) {
                    items = [
                        FALLBACK
                    ];
                    mapped[0] = createRoot((disposer)=>{
                        disposers[0] = disposer;
                        return options.fallback();
                    });
                    len = 1;
                }
                return mapped;
            }
            if (items[0] === FALLBACK) {
                disposers[0]();
                disposers = [];
                items = [];
                mapped = [];
                len = 0;
            }
            for(i = 0; i < newItems.length; i++){
                if (i < items.length && items[i] !== newItems[i]) signals[i](()=>newItems[i]);
                else if (i >= items.length) mapped[i] = createRoot(mapper);
            }
            for(; i < items.length; i++)disposers[i]();
            len = signals.length = disposers.length = newItems.length;
            items = newItems.slice(0);
            return mapped = mapped.slice(0, len);
        });
        function mapper(disposer) {
            disposers[i] = disposer;
            const [s, set] = createSignal(newItems[i]);
            signals[i] = set;
            return mapFn(s, i);
        }
    };
}
let hydrationEnabled = false;
function enableHydration() {
    hydrationEnabled = true;
}
function createComponent(Comp, props) {
    if (hydrationEnabled) {
        if (sharedConfig.context) {
            const c = sharedConfig.context;
            setHydrateContext(nextHydrateContext());
            const r = untrack(()=>Comp(props || {}));
            setHydrateContext(c);
            return r;
        }
    }
    return untrack(()=>Comp(props || {}));
}
function trueFn() {
    return true;
}
const propTraps = {
    get (_, property, receiver) {
        if (property === $PROXY) return receiver;
        return _.get(property);
    },
    has (_, property) {
        if (property === $PROXY) return true;
        return _.has(property);
    },
    set: trueFn,
    deleteProperty: trueFn,
    getOwnPropertyDescriptor (_, property) {
        return {
            configurable: true,
            enumerable: true,
            get () {
                return _.get(property);
            },
            set: trueFn,
            deleteProperty: trueFn
        };
    },
    ownKeys (_) {
        return _.keys();
    }
};
function resolveSource(s) {
    return !(s = typeof s === "function" ? s() : s) ? {} : s;
}
function resolveSources() {
    for(let i = 0, length = this.length; i < length; ++i){
        const v = this[i]();
        if (v !== undefined) return v;
    }
}
function mergeProps(...sources) {
    let proxy = false;
    for(let i = 0; i < sources.length; i++){
        const s = sources[i];
        proxy = proxy || !!s && $PROXY in s;
        sources[i] = typeof s === "function" ? (proxy = true, createMemo(s)) : s;
    }
    if (proxy) return new Proxy({
        get (property) {
            for(let i = sources.length - 1; i >= 0; i--){
                const v = resolveSource(sources[i])[property];
                if (v !== undefined) return v;
            }
        },
        has (property) {
            for(let i = sources.length - 1; i >= 0; i--){
                if (property in resolveSource(sources[i])) return true;
            }
            return false;
        },
        keys () {
            const keys = [];
            for(let i = 0; i < sources.length; i++)keys.push(...Object.keys(resolveSource(sources[i])));
            return [
                ...new Set(keys)
            ];
        }
    }, propTraps);
    const sourcesMap = {};
    const defined = Object.create(null);
    for(let i = sources.length - 1; i >= 0; i--){
        const source = sources[i];
        if (!source) continue;
        const sourceKeys = Object.getOwnPropertyNames(source);
        for(let i = sourceKeys.length - 1; i >= 0; i--){
            const key = sourceKeys[i];
            if (key === "__proto__" || key === "constructor") continue;
            const desc = Object.getOwnPropertyDescriptor(source, key);
            if (!defined[key]) defined[key] = desc.get ? {
                enumerable: true,
                configurable: true,
                get: resolveSources.bind(sourcesMap[key] = [
                    desc.get.bind(source)
                ])
            } : desc.value !== undefined ? desc : undefined;
            else {
                const sources = sourcesMap[key];
                if (sources) {
                    if (desc.get) sources.push(desc.get.bind(source));
                    else if (desc.value !== undefined) sources.push(()=>desc.value);
                }
            }
        }
    }
    const target = {};
    const definedKeys = Object.keys(defined);
    for(let i = definedKeys.length - 1; i >= 0; i--){
        const key = definedKeys[i], desc = defined[key];
        if (desc && desc.get) Object.defineProperty(target, key, desc);
        else target[key] = desc ? desc.value : undefined;
    }
    return target;
}
function splitProps(props, ...keys) {
    if ($PROXY in props) {
        const blocked = new Set(keys.length > 1 ? keys.flat() : keys[0]);
        const res = keys.map((k)=>{
            return new Proxy({
                get (property) {
                    return k.includes(property) ? props[property] : undefined;
                },
                has (property) {
                    return k.includes(property) && property in props;
                },
                keys () {
                    return k.filter((property)=>property in props);
                }
            }, propTraps);
        });
        res.push(new Proxy({
            get (property) {
                return blocked.has(property) ? undefined : props[property];
            },
            has (property) {
                return blocked.has(property) ? false : property in props;
            },
            keys () {
                return Object.keys(props).filter((k)=>!blocked.has(k));
            }
        }, propTraps));
        return res;
    }
    const otherObject = {};
    const objects = keys.map(()=>({}));
    for (const propName of Object.getOwnPropertyNames(props)){
        const desc = Object.getOwnPropertyDescriptor(props, propName);
        const isDefaultDesc = !desc.get && !desc.set && desc.enumerable && desc.writable && desc.configurable;
        let blocked = false;
        let objectIndex = 0;
        for (const k of keys){
            if (k.includes(propName)) {
                blocked = true;
                isDefaultDesc ? objects[objectIndex][propName] = desc.value : Object.defineProperty(objects[objectIndex], propName, desc);
            }
            ++objectIndex;
        }
        if (!blocked) isDefaultDesc ? otherObject[propName] = desc.value : Object.defineProperty(otherObject, propName, desc);
    }
    return [
        ...objects,
        otherObject
    ];
}
function lazy(fn) {
    let comp;
    let p;
    const wrap = (props)=>{
        const ctx = sharedConfig.context;
        if (ctx) {
            const [s, set] = createSignal();
            sharedConfig.count || (sharedConfig.count = 0);
            sharedConfig.count++;
            (p || (p = fn())).then((mod)=>{
                setHydrateContext(ctx);
                sharedConfig.count--;
                set(()=>mod.default);
                setHydrateContext();
            });
            comp = s;
        } else if (!comp) {
            const [s] = createResource(()=>(p || (p = fn())).then((mod)=>mod.default));
            comp = s;
        }
        let Comp;
        return createMemo(()=>(Comp = comp()) && untrack(()=>{
                if (!ctx) return Comp(props);
                const c = sharedConfig.context;
                setHydrateContext(ctx);
                const r = Comp(props);
                setHydrateContext(c);
                return r;
            }));
    };
    wrap.preload = ()=>p || ((p = fn()).then((mod)=>comp = ()=>mod.default), p);
    return wrap;
}
let counter = 0;
function createUniqueId() {
    const ctx = sharedConfig.context;
    return ctx ? `${ctx.id}${ctx.count++}` : `cl-${counter++}`;
}
const narrowedError = (name)=>`Stale read from <${name}>.`;
function For(props) {
    const fallback = "fallback" in props && {
        fallback: ()=>props.fallback
    };
    return createMemo(mapArray(()=>props.each, props.children, fallback || undefined));
}
function Index(props) {
    const fallback = "fallback" in props && {
        fallback: ()=>props.fallback
    };
    return createMemo(indexArray(()=>props.each, props.children, fallback || undefined));
}
function Show(props) {
    const keyed = props.keyed;
    const condition = createMemo(()=>props.when, undefined, {
        equals: (a, b)=>keyed ? a === b : !a === !b
    });
    return createMemo(()=>{
        const c = condition();
        if (c) {
            const child = props.children;
            const fn = typeof child === "function" && child.length > 0;
            return fn ? untrack(()=>child(keyed ? c : ()=>{
                    if (!untrack(condition)) throw narrowedError("Show");
                    return props.when;
                })) : child;
        }
        return props.fallback;
    }, undefined, undefined);
}
function Switch(props) {
    let keyed = false;
    const equals = (a, b)=>(keyed ? a[1] === b[1] : !a[1] === !b[1]) && a[2] === b[2];
    const conditions = children(()=>props.children), evalConditions = createMemo(()=>{
        let conds = conditions();
        if (!Array.isArray(conds)) conds = [
            conds
        ];
        for(let i = 0; i < conds.length; i++){
            const c = conds[i].when;
            if (c) {
                keyed = !!conds[i].keyed;
                return [
                    i,
                    c,
                    conds[i]
                ];
            }
        }
        return [
            -1
        ];
    }, undefined, {
        equals
    });
    return createMemo(()=>{
        const [index, when, cond] = evalConditions();
        if (index < 0) return props.fallback;
        const c = cond.children;
        const fn = typeof c === "function" && c.length > 0;
        return fn ? untrack(()=>c(keyed ? when : ()=>{
                if (untrack(evalConditions)[0] !== index) throw narrowedError("Match");
                return cond.when;
            })) : c;
    }, undefined, undefined);
}
function Match(props) {
    return props;
}
let Errors;
function resetErrorBoundaries() {
    Errors && [
        ...Errors
    ].forEach((fn)=>fn());
}
function ErrorBoundary(props) {
    let err;
    if (sharedConfig.context && sharedConfig.load) err = sharedConfig.load(sharedConfig.context.id + sharedConfig.context.count);
    const [errored, setErrored] = createSignal(err, undefined);
    Errors || (Errors = new Set());
    Errors.add(setErrored);
    onCleanup(()=>Errors.delete(setErrored));
    return createMemo(()=>{
        let e;
        if (e = errored()) {
            const f = props.fallback;
            return typeof f === "function" && f.length ? untrack(()=>f(e, ()=>setErrored())) : f;
        }
        return catchError(()=>props.children, setErrored);
    }, undefined, undefined);
}
const suspenseListEquals = (a, b)=>a.showContent === b.showContent && a.showFallback === b.showFallback;
const SuspenseListContext = createContext();
function SuspenseList(props) {
    let [wrapper, setWrapper] = createSignal(()=>({
            inFallback: false
        })), show;
    const listContext = useContext(SuspenseListContext);
    const [registry, setRegistry] = createSignal([]);
    if (listContext) show = listContext.register(createMemo(()=>wrapper()().inFallback));
    const resolved = createMemo((prev)=>{
        const reveal = props.revealOrder, tail = props.tail, { showContent = true, showFallback = true } = show ? show() : {}, reg = registry(), reverse = reveal === "backwards";
        if (reveal === "together") {
            const all = reg.every((inFallback)=>!inFallback());
            const res = reg.map(()=>({
                    showContent: all && showContent,
                    showFallback
                }));
            res.inFallback = !all;
            return res;
        }
        let stop = false;
        let inFallback = prev.inFallback;
        const res = [];
        for(let i = 0, len = reg.length; i < len; i++){
            const n = reverse ? len - i - 1 : i, s = reg[n]();
            if (!stop && !s) res[n] = {
                showContent,
                showFallback
            };
            else {
                const next = !stop;
                if (next) inFallback = true;
                res[n] = {
                    showContent: next,
                    showFallback: !tail || next && tail === "collapsed" ? showFallback : false
                };
                stop = true;
            }
        }
        if (!stop) inFallback = false;
        res.inFallback = inFallback;
        return res;
    }, {
        inFallback: false
    });
    setWrapper(()=>resolved);
    return createComponent(SuspenseListContext.Provider, {
        value: {
            register: (inFallback)=>{
                let index;
                setRegistry((registry)=>{
                    index = registry.length;
                    return [
                        ...registry,
                        inFallback
                    ];
                });
                return createMemo(()=>resolved()[index], undefined, {
                    equals: suspenseListEquals
                });
            }
        },
        get children () {
            return props.children;
        }
    });
}
function Suspense(props) {
    let counter = 0, show, ctx, p, flicker, error;
    const [inFallback, setFallback] = createSignal(false), SuspenseContext = getSuspenseContext(), store = {
        increment: ()=>{
            if (++counter === 1) setFallback(true);
        },
        decrement: ()=>{
            if (--counter === 0) setFallback(false);
        },
        inFallback,
        effects: [],
        resolved: false
    }, owner = getOwner();
    if (sharedConfig.context && sharedConfig.load) {
        const key = sharedConfig.context.id + sharedConfig.context.count;
        let ref = sharedConfig.load(key);
        if (ref) {
            if (typeof ref !== "object" || ref.status !== "success") p = ref;
            else sharedConfig.gather(key);
        }
        if (p && p !== "$$f") {
            const [s, set] = createSignal(undefined, {
                equals: false
            });
            flicker = s;
            p.then(()=>{
                if (sharedConfig.done) return set();
                sharedConfig.gather(key);
                setHydrateContext(ctx);
                set();
                setHydrateContext();
            }, (err)=>{
                error = err;
                set();
            });
        }
    }
    const listContext = useContext(SuspenseListContext);
    if (listContext) show = listContext.register(store.inFallback);
    let dispose;
    onCleanup(()=>dispose && dispose());
    return createComponent(SuspenseContext.Provider, {
        value: store,
        get children () {
            return createMemo(()=>{
                if (error) throw error;
                ctx = sharedConfig.context;
                if (flicker) {
                    flicker();
                    return flicker = undefined;
                }
                if (ctx && p === "$$f") setHydrateContext();
                const rendered = createMemo(()=>props.children);
                return createMemo((prev)=>{
                    const inFallback = store.inFallback(), { showContent = true, showFallback = true } = show ? show() : {};
                    if ((!inFallback || p && p !== "$$f") && showContent) {
                        store.resolved = true;
                        dispose && dispose();
                        dispose = ctx = p = undefined;
                        resumeEffects(store.effects);
                        return rendered();
                    }
                    if (!showFallback) return;
                    if (dispose) return prev;
                    return createRoot((disposer)=>{
                        dispose = disposer;
                        if (ctx) {
                            setHydrateContext({
                                id: ctx.id + "f",
                                count: 0
                            });
                            ctx = undefined;
                        }
                        return props.fallback;
                    }, owner);
                });
            });
        }
    });
}
const DEV = undefined;

}),
"683": (function (__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {
"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony import */var _widget_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./widget.js */"30");

// @ts-expect-error -- define is a global provided by the notebook runtime.
define([
    "@jupyter-widgets/base"
], _widget_js__WEBPACK_IMPORTED_MODULE_0__["default"]);
}),
"30": (function (__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {
"use strict";
__webpack_require__.r(__webpack_exports__);
__webpack_require__.d(__webpack_exports__, {
  "default": function() { return /* export default binding */ __WEBPACK_DEFAULT_EXPORT__; }
});
/* harmony import */var _lukeed_uuid__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @lukeed/uuid */"800");
/* harmony import */var solid_js__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! solid-js */"336");


/**
 * @typedef AnyWidget
 * @prop initialize {import("@anywidget/types").Initialize}
 * @prop render {import("@anywidget/types").Render}
 */ /**
 *  @typedef AnyWidgetModule
 *  @prop render {import("@anywidget/types").Render=}
 *  @prop default {AnyWidget | (() => AnyWidget | Promise<AnyWidget>)=}
 */ /**
 * @param {any} condition
 * @param {string} message
 * @returns {asserts condition}
 */ function assert(condition, message) {
    if (!condition) throw new Error(message);
}
/**
 * @param {string} str
 * @returns {str is "https://${string}" | "http://${string}"}
 */ function is_href(str) {
    return str.startsWith("http://") || str.startsWith("https://");
}
/**
 * @param {string} href
 * @param {string} anywidget_id
 * @returns {Promise<void>}
 */ async function load_css_href(href, anywidget_id) {
    /** @type {HTMLLinkElement | null} */ let prev = document.querySelector(`link[id='${anywidget_id}']`);
    // Adapted from https://github.com/vitejs/vite/blob/d59e1acc2efc0307488364e9f2fad528ec57f204/packages/vite/src/client/client.ts#L185-L201
    // Swaps out old styles with new, but avoids flash of unstyled content.
    // No need to await the load since we already have styles applied.
    if (prev) {
        let newLink = /** @type {HTMLLinkElement} */ prev.cloneNode();
        newLink.href = href;
        newLink.addEventListener("load", ()=>prev?.remove());
        newLink.addEventListener("error", ()=>prev?.remove());
        prev.after(newLink);
        return;
    }
    return new Promise((resolve)=>{
        let link = Object.assign(document.createElement("link"), {
            rel: "stylesheet",
            href,
            onload: resolve
        });
        document.head.appendChild(link);
    });
}
/**
 * @param {string} css_text
 * @param {string} anywidget_id
 * @returns {void}
 */ function load_css_text(css_text, anywidget_id) {
    /** @type {HTMLStyleElement | null} */ let prev = document.querySelector(`style[id='${anywidget_id}']`);
    if (prev) {
        // replace instead of creating a new DOM node
        prev.textContent = css_text;
        return;
    }
    let style = Object.assign(document.createElement("style"), {
        id: anywidget_id,
        type: "text/css"
    });
    style.appendChild(document.createTextNode(css_text));
    document.head.appendChild(style);
}
/**
 * @param {string | undefined} css
 * @param {string} anywidget_id
 * @returns {Promise<void>}
 */ async function load_css(css, anywidget_id) {
    if (!css || !anywidget_id) return;
    if (is_href(css)) return load_css_href(css, anywidget_id);
    return load_css_text(css, anywidget_id);
}
/**
 * @param {string} esm
 * @returns {Promise<{ mod: AnyWidgetModule, url: string }>}
 */ async function load_esm(esm) {
    if (is_href(esm)) return {
        mod: await import(/* webpackIgnore: true */ esm),
        url: esm
    };
    let url = URL.createObjectURL(new Blob([
        esm
    ], {
        type: "text/javascript"
    }));
    let mod = await import(/* webpackIgnore: true */ url);
    URL.revokeObjectURL(url);
    return {
        mod,
        url
    };
}
function warn_render_deprecation() {
    console.warn(`\
[anywidget] Deprecation Warning. Direct export of a 'render' will likely be deprecated in the future. To migrate ...

Remove the 'export' keyword from 'render'
-----------------------------------------

export function render({ model, el }) { ... }
^^^^^^

Create a default export that returns an object with 'render'
------------------------------------------------------------

function render({ model, el }) { ... }
         ^^^^^^
export default { render }
                 ^^^^^^

To learn more, please see: https://github.com/manzt/anywidget/pull/395
`);
}
/**
 * @param {string} esm
 * @returns {Promise<AnyWidget & { url: string }>}
 */ async function load_widget(esm) {
    let { mod, url } = await load_esm(esm);
    if (mod.render) {
        warn_render_deprecation();
        return {
            url,
            async initialize () {},
            render: mod.render
        };
    }
    assert(mod.default, `[anywidget] module must export a default function or object.`);
    let widget = typeof mod.default === "function" ? await mod.default() : mod.default;
    return {
        url,
        ...widget
    };
}
/**
 * This is a trick so that we can cleanup event listeners added
 * by the user-defined function.
 */ let INITIALIZE_MARKER = Symbol("anywidget.initialize");
/**
 * @param {import("@jupyter-widgets/base").DOMWidgetModel} model
 * @param {unknown} context
 * @return {import("@anywidget/types").AnyModel}
 *
 * Prunes the view down to the minimum context necessary.
 *
 * Calls to `model.get` and `model.set` automatically add the
 * `context`, so we can gracefully unsubscribe from events
 * added by user-defined hooks.
 */ function model_proxy(model, context) {
    return {
        get: model.get.bind(model),
        set: model.set.bind(model),
        save_changes: model.save_changes.bind(model),
        send: model.send.bind(model),
        // @ts-expect-error
        on (name, callback) {
            model.on(name, callback, context);
        },
        off (name, callback) {
            model.off(name, callback, context);
        },
        widget_manager: model.widget_manager
    };
}
/**
 * @param {void | (() => import('vitest').Awaitable<void>)} fn
 * @param {string} kind
 */ async function safe_cleanup(fn, kind) {
    return Promise.resolve().then(()=>fn?.()).catch((e)=>console.warn(`[anywidget] error cleaning up ${kind}.`, e));
}
/**
 * @template T
 * @typedef {{ data: T, state: "ok" } | { error: any, state: "error" }} Result
 */ /** @type {<T>(data: T) => Result<T>} */ function ok(data) {
    return {
        data,
        state: "ok"
    };
}
/** @type {(e: any) => Result<any>} */ function error(e) {
    return {
        error: e,
        state: "error"
    };
}
/**
 * Cleans up the stack trace at anywidget boundary.
 * You can fully inspect the entire stack trace in the console interactively,
 * but the initial error message is cleaned up to be more user-friendly.
 *
 * @param {unknown} source
 * @returns {never}
 */ function throw_anywidget_error(source) {
    if (!(source instanceof Error)) // Don't know what to do with this.
    throw source;
    let lines = source.stack?.split("\n") ?? [];
    let anywidget_index = lines.findIndex((line)=>line.includes("anywidget"));
    let clean_stack = anywidget_index === -1 ? lines : lines.slice(0, anywidget_index + 1);
    source.stack = clean_stack.join("\n");
    console.error(source);
    throw source;
}
/**
 * @typedef InvokeOptions
 * @prop {DataView[]} [buffers]
 * @prop {AbortSignal} [signal]
 */ /**
 * @template T
 * @param {import("@anywidget/types").AnyModel} model
 * @param {string} name
 * @param {any} [msg]
 * @param {InvokeOptions} [options]
 * @return {Promise<[T, DataView[]]>}
 */ function invoke(model, name, msg, options = {}) {
    // crypto.randomUUID() is not available in non-secure contexts (i.e., http://)
    // so we use simple (non-secure) polyfill.
    let id = _lukeed_uuid__WEBPACK_IMPORTED_MODULE_0__.v4();
    let signal = options.signal ?? AbortSignal.timeout(3000);
    return new Promise((resolve, reject)=>{
        if (signal.aborted) reject(signal.reason);
        signal.addEventListener("abort", ()=>{
            model.off("msg:custom", handler);
            reject(signal.reason);
        });
        /**
		 * @param {{ id: string, kind: "anywidget-command-response", response: T }} msg
		 * @param {DataView[]} buffers
		 */ function handler(msg, buffers) {
            if (!(msg.id === id)) return;
            resolve([
                msg.response,
                buffers
            ]);
            model.off("msg:custom", handler);
        }
        model.on("msg:custom", handler);
        model.send({
            id,
            kind: "anywidget-command",
            name,
            msg
        }, undefined, options.buffers ?? []);
    });
}
class Runtime {
    /** @type {() => void} */ #disposer = ()=>{};
    /** @type {Set<() => void>} */ #view_disposers = new Set();
    /** @type {import('solid-js').Resource<Result<AnyWidget & { url: string }>>} */ // @ts-expect-error - Set synchronously in constructor.
    #widget_result;
    /** @param {import("@jupyter-widgets/base").DOMWidgetModel} model */ constructor(model){
        this.#disposer = solid_js__WEBPACK_IMPORTED_MODULE_1__.createRoot((dispose)=>{
            let [css, set_css] = solid_js__WEBPACK_IMPORTED_MODULE_1__.createSignal(model.get("_css"));
            model.on("change:_css", ()=>{
                let id = model.get("_anywidget_id");
                console.debug(`[anywidget] css hot updated: ${id}`);
                set_css(model.get("_css"));
            });
            solid_js__WEBPACK_IMPORTED_MODULE_1__.createEffect(()=>{
                let id = model.get("_anywidget_id");
                load_css(css(), id);
            });
            /** @type {import("solid-js").Signal<string>} */ let [esm, setEsm] = solid_js__WEBPACK_IMPORTED_MODULE_1__.createSignal(model.get("_esm"));
            model.on("change:_esm", async ()=>{
                let id = model.get("_anywidget_id");
                console.debug(`[anywidget] esm hot updated: ${id}`);
                setEsm(model.get("_esm"));
            });
            /** @type {void | (() => import("vitest").Awaitable<void>)} */ let cleanup;
            this.#widget_result = solid_js__WEBPACK_IMPORTED_MODULE_1__.createResource(esm, async (update)=>{
                await safe_cleanup(cleanup, "initialize");
                try {
                    model.off(null, null, INITIALIZE_MARKER);
                    let widget = await load_widget(update);
                    cleanup = await widget.initialize?.({
                        model: model_proxy(model, INITIALIZE_MARKER),
                        experimental: {
                            // @ts-expect-error - bind isn't working
                            invoke: invoke.bind(null, model)
                        }
                    });
                    return ok(widget);
                } catch (e) {
                    return error(e);
                }
            })[0];
            return ()=>{
                cleanup?.();
                model.off("change:_css");
                model.off("change:_esm");
                dispose();
            };
        });
    }
    /**
	 * @param {import("@jupyter-widgets/base").DOMWidgetView} view
	 * @returns {Promise<() => void>}
	 */ async create_view(view) {
        let model = view.model;
        let disposer = solid_js__WEBPACK_IMPORTED_MODULE_1__.createRoot((dispose)=>{
            /** @type {void | (() => import("vitest").Awaitable<void>)} */ let cleanup;
            let resource = solid_js__WEBPACK_IMPORTED_MODULE_1__.createResource(this.#widget_result, async (widget_result)=>{
                cleanup?.();
                // Clear all previous event listeners from this hook.
                model.off(null, null, view);
                view.$el.empty();
                if (widget_result.state === "error") throw_anywidget_error(widget_result.error);
                let widget = widget_result.data;
                try {
                    cleanup = await widget.render?.({
                        model: model_proxy(model, view),
                        el: view.el,
                        experimental: {
                            // @ts-expect-error - bind isn't working
                            invoke: invoke.bind(null, model)
                        }
                    });
                } catch (e) {
                    throw_anywidget_error(e);
                }
            })[0];
            solid_js__WEBPACK_IMPORTED_MODULE_1__.createEffect(()=>{
                resource.error;
            });
            return ()=>{
                dispose();
                cleanup?.();
            };
        });
        // Have the runtime keep track but allow the view to dispose itself.
        this.#view_disposers.add(disposer);
        return ()=>{
            let deleted = this.#view_disposers.delete(disposer);
            if (deleted) disposer();
        };
    }
    dispose() {
        this.#view_disposers.forEach((dispose)=>dispose());
        this.#view_disposers.clear();
        this.#disposer();
    }
}
// @ts-expect-error - injected by bundler
let version = "0.9.9";
/** @param {typeof import("@jupyter-widgets/base")} base */ /* harmony default export */ function __WEBPACK_DEFAULT_EXPORT__({ DOMWidgetModel, DOMWidgetView }) {
    /** @type {WeakMap<AnyModel, Runtime>} */ let RUNTIMES = new WeakMap();
    class AnyModel extends DOMWidgetModel {
        static model_name = "AnyModel";
        static model_module = "anywidget";
        static model_module_version = version;
        static view_name = "AnyView";
        static view_module = "anywidget";
        static view_module_version = version;
        /** @param {Parameters<InstanceType<DOMWidgetModel>["initialize"]>} args */ initialize(...args) {
            super.initialize(...args);
            let runtime = new Runtime(this);
            this.once("destroy", ()=>{
                try {
                    runtime.dispose();
                } finally{
                    RUNTIMES.delete(this);
                }
            });
            RUNTIMES.set(this, runtime);
        }
        /**
		 * @param {Record<string, any>} state
		 *
		 * We override to support binary trailets because JSON.parse(JSON.stringify())
		 * does not properly clone binary data (it just returns an empty object).
		 *
		 * https://github.com/jupyter-widgets/ipywidgets/blob/47058a373d2c2b3acf101677b2745e14b76dd74b/packages/base/src/widget.ts#L562-L583
		 */ serialize(state) {
            let serializers = /** @type {DOMWidgetModel} */ this.constructor.serializers || {};
            for (let k of Object.keys(state))try {
                let serialize = serializers[k]?.serialize;
                if (serialize) state[k] = serialize(state[k], this);
                else if (k === "layout" || k === "style") // These keys come from ipywidgets, rely on JSON.stringify trick.
                state[k] = JSON.parse(JSON.stringify(state[k]));
                else state[k] = structuredClone(state[k]);
                if (typeof state[k]?.toJSON === "function") state[k] = state[k].toJSON();
            } catch (e) {
                console.error("Error serializing widget state attribute: ", k);
                throw e;
            }
            return state;
        }
    }
    class AnyView extends DOMWidgetView {
        /** @type {undefined | (() => void)} */ #dispose = undefined;
        async render() {
            let runtime = RUNTIMES.get(this.model);
            assert(runtime, "[anywidget] runtime not found.");
            assert(!this.#dispose, "[anywidget] dispose already set.");
            this.#dispose = await runtime.create_view(this);
        }
        remove() {
            this.#dispose?.();
            super.remove();
        }
    }
    return {
        AnyModel,
        AnyView
    };
}
}),

}
// The module cache
 var __webpack_module_cache__ = {};
function __webpack_require__(moduleId) {
// Check if module is in cache
        var cachedModule = __webpack_module_cache__[moduleId];
        if (cachedModule !== undefined) {
      return cachedModule.exports;
      }
      // Create a new module (and put it into the cache)
      var module = (__webpack_module_cache__[moduleId] = {
       exports: {}
      });
      // Execute the module function
      __webpack_modules__[moduleId](module, module.exports, __webpack_require__);
// Return the exports of the module
 return module.exports;

}
// expose the modules object (__webpack_modules__)
 __webpack_require__.m = __webpack_modules__;
// expose the module cache
 __webpack_require__.c = __webpack_module_cache__;
// webpack/runtime/define_property_getters
!function() {
__webpack_require__.d = function(exports, definition) {
	for(var key in definition) {
        if(__webpack_require__.o(definition, key) && !__webpack_require__.o(exports, key)) {
            Object.defineProperty(exports, key, { enumerable: true, get: definition[key] });
        }
    }
};
}();
// webpack/runtime/has_own_property
!function() {
__webpack_require__.o = function (obj, prop) {
	return Object.prototype.hasOwnProperty.call(obj, prop);
};

}();
// webpack/runtime/make_namespace_object
!function() {
// define __esModule on exports
__webpack_require__.r = function(exports) {
	if(typeof Symbol !== 'undefined' && Symbol.toStringTag) {
		Object.defineProperty(exports, Symbol.toStringTag, { value: 'Module' });
	}
	Object.defineProperty(exports, '__esModule', { value: true });
};

}();
// webpack/runtime/sharing
!function() {

__webpack_require__.S = {};
__webpack_require__.initializeSharingData = { scopeToSharingDataMapping: {  }, uniqueName: "@anywidget/monorepo" };
var initPromises = {};
var initTokens = {};
__webpack_require__.I = function(name, initScope) {
	if (!initScope) initScope = [];
	// handling circular init calls
	var initToken = initTokens[name];
	if (!initToken) initToken = initTokens[name] = {};
	if (initScope.indexOf(initToken) >= 0) return;
	initScope.push(initToken);
	// only runs once
	if (initPromises[name]) return initPromises[name];
	// creates a new share scope if needed
	if (!__webpack_require__.o(__webpack_require__.S, name))
		__webpack_require__.S[name] = {};
	// runs all init snippets from all modules reachable
	var scope = __webpack_require__.S[name];
	var warn = function (msg) {
		if (typeof console !== "undefined" && console.warn) console.warn(msg);
	};
	var uniqueName = __webpack_require__.initializeSharingData.uniqueName;
	var register = function (name, version, factory, eager) {
		var versions = (scope[name] = scope[name] || {});
		var activeVersion = versions[version];
		if (
			!activeVersion ||
			(!activeVersion.loaded &&
				(!eager != !activeVersion.eager
					? eager
					: uniqueName > activeVersion.from))
		)
			versions[version] = { get: factory, from: uniqueName, eager: !!eager };
	};
	var initExternal = function (id) {
		var handleError = function (err) {
			warn("Initialization of sharing external failed: " + err);
		};
		try {
			var module = __webpack_require__(id);
			if (!module) return;
			var initFn = function (module) {
				return (
					module &&
					module.init &&
					module.init(__webpack_require__.S[name], initScope)
				);
			};
			if (module.then) return promises.push(module.then(initFn, handleError));
			var initResult = initFn(module);
			if (initResult && initResult.then)
				return promises.push(initResult["catch"](handleError));
		} catch (err) {
			handleError(err);
		}
	};
	var promises = [];
	var scopeToSharingDataMapping = __webpack_require__.initializeSharingData.scopeToSharingDataMapping;
	if (scopeToSharingDataMapping[name]) {
		scopeToSharingDataMapping[name].forEach(function (stage) {
			if (typeof stage === "object") register(stage.name, stage.version, stage.factory, stage.eager);
			else initExternal(stage)
		});
	}
	if (!promises.length) return (initPromises[name] = 1);
	return (initPromises[name] = Promise.all(promises).then(function () {
		return (initPromises[name] = 1);
	}));
};


}();
var __webpack_exports__ = __webpack_require__("683");
})()

//# sourceMappingURL=main.77fc3747.js.map