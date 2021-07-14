/******/ (function(modules) { // webpackBootstrap
/******/ 	// The module cache
/******/ 	var installedModules = {};
/******/
/******/ 	// The require function
/******/ 	function __webpack_require__(moduleId) {
/******/
/******/ 		// Check if module is in cache
/******/ 		if(installedModules[moduleId]) {
/******/ 			return installedModules[moduleId].exports;
/******/ 		}
/******/ 		// Create a new module (and put it into the cache)
/******/ 		var module = installedModules[moduleId] = {
/******/ 			i: moduleId,
/******/ 			l: false,
/******/ 			exports: {}
/******/ 		};
/******/
/******/ 		// Execute the module function
/******/ 		modules[moduleId].call(module.exports, module, module.exports, __webpack_require__);
/******/
/******/ 		// Flag the module as loaded
/******/ 		module.l = true;
/******/
/******/ 		// Return the exports of the module
/******/ 		return module.exports;
/******/ 	}
/******/
/******/
/******/ 	// expose the modules object (__webpack_modules__)
/******/ 	__webpack_require__.m = modules;
/******/
/******/ 	// expose the module cache
/******/ 	__webpack_require__.c = installedModules;
/******/
/******/ 	// define getter function for harmony exports
/******/ 	__webpack_require__.d = function(exports, name, getter) {
/******/ 		if(!__webpack_require__.o(exports, name)) {
/******/ 			Object.defineProperty(exports, name, { enumerable: true, get: getter });
/******/ 		}
/******/ 	};
/******/
/******/ 	// define __esModule on exports
/******/ 	__webpack_require__.r = function(exports) {
/******/ 		if(typeof Symbol !== 'undefined' && Symbol.toStringTag) {
/******/ 			Object.defineProperty(exports, Symbol.toStringTag, { value: 'Module' });
/******/ 		}
/******/ 		Object.defineProperty(exports, '__esModule', { value: true });
/******/ 	};
/******/
/******/ 	// create a fake namespace object
/******/ 	// mode & 1: value is a module id, require it
/******/ 	// mode & 2: merge all properties of value into the ns
/******/ 	// mode & 4: return value when already ns object
/******/ 	// mode & 8|1: behave like require
/******/ 	__webpack_require__.t = function(value, mode) {
/******/ 		if(mode & 1) value = __webpack_require__(value);
/******/ 		if(mode & 8) return value;
/******/ 		if((mode & 4) && typeof value === 'object' && value && value.__esModule) return value;
/******/ 		var ns = Object.create(null);
/******/ 		__webpack_require__.r(ns);
/******/ 		Object.defineProperty(ns, 'default', { enumerable: true, value: value });
/******/ 		if(mode & 2 && typeof value != 'string') for(var key in value) __webpack_require__.d(ns, key, function(key) { return value[key]; }.bind(null, key));
/******/ 		return ns;
/******/ 	};
/******/
/******/ 	// getDefaultExport function for compatibility with non-harmony modules
/******/ 	__webpack_require__.n = function(module) {
/******/ 		var getter = module && module.__esModule ?
/******/ 			function getDefault() { return module['default']; } :
/******/ 			function getModuleExports() { return module; };
/******/ 		__webpack_require__.d(getter, 'a', getter);
/******/ 		return getter;
/******/ 	};
/******/
/******/ 	// Object.prototype.hasOwnProperty.call
/******/ 	__webpack_require__.o = function(object, property) { return Object.prototype.hasOwnProperty.call(object, property); };
/******/
/******/ 	// __webpack_public_path__
/******/ 	__webpack_require__.p = "";
/******/
/******/
/******/ 	// Load entry module and return exports
/******/ 	return __webpack_require__(__webpack_require__.s = "./src/main/js/PageTitleComponent.js");
/******/ })
/************************************************************************/
/******/ ({

/***/ "./src/main/js/PageTitleComponent.js":
/***/ (function(module, exports) {

function _typeof(obj) { "@babel/helpers - typeof"; if (typeof Symbol === "function" && typeof Symbol.iterator === "symbol") { _typeof = function _typeof(obj) { return typeof obj; }; } else { _typeof = function _typeof(obj) { return obj && typeof Symbol === "function" && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }; } return _typeof(obj); }

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

function _defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } }

function _createClass(Constructor, protoProps, staticProps) { if (protoProps) _defineProperties(Constructor.prototype, protoProps); if (staticProps) _defineProperties(Constructor, staticProps); return Constructor; }

function _createSuper(Derived) { return function () { var Super = _getPrototypeOf(Derived), result; if (_isNativeReflectConstruct()) { var NewTarget = _getPrototypeOf(this).constructor; result = Reflect.construct(Super, arguments, NewTarget); } else { result = Super.apply(this, arguments); } return _possibleConstructorReturn(this, result); }; }

function _possibleConstructorReturn(self, call) { if (call && (_typeof(call) === "object" || typeof call === "function")) { return call; } return _assertThisInitialized(self); }

function _assertThisInitialized(self) { if (self === void 0) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return self; }

function _inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function"); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, writable: true, configurable: true } }); if (superClass) _setPrototypeOf(subClass, superClass); }

function _wrapNativeSuper(Class) { var _cache = typeof Map === "function" ? new Map() : undefined; _wrapNativeSuper = function _wrapNativeSuper(Class) { if (Class === null || !_isNativeFunction(Class)) return Class; if (typeof Class !== "function") { throw new TypeError("Super expression must either be null or a function"); } if (typeof _cache !== "undefined") { if (_cache.has(Class)) return _cache.get(Class); _cache.set(Class, Wrapper); } function Wrapper() { return _construct(Class, arguments, _getPrototypeOf(this).constructor); } Wrapper.prototype = Object.create(Class.prototype, { constructor: { value: Wrapper, enumerable: false, writable: true, configurable: true } }); return _setPrototypeOf(Wrapper, Class); }; return _wrapNativeSuper(Class); }

function _construct(Parent, args, Class) { if (_isNativeReflectConstruct()) { _construct = Reflect.construct; } else { _construct = function _construct(Parent, args, Class) { var a = [null]; a.push.apply(a, args); var Constructor = Function.bind.apply(Parent, a); var instance = new Constructor(); if (Class) _setPrototypeOf(instance, Class.prototype); return instance; }; } return _construct.apply(null, arguments); }

function _isNativeReflectConstruct() { if (typeof Reflect === "undefined" || !Reflect.construct) return false; if (Reflect.construct.sham) return false; if (typeof Proxy === "function") return true; try { Date.prototype.toString.call(Reflect.construct(Date, [], function () {})); return true; } catch (e) { return false; } }

function _isNativeFunction(fn) { return Function.toString.call(fn).indexOf("[native code]") !== -1; }

function _setPrototypeOf(o, p) { _setPrototypeOf = Object.setPrototypeOf || function _setPrototypeOf(o, p) { o.__proto__ = p; return o; }; return _setPrototypeOf(o, p); }

function _getPrototypeOf(o) { _getPrototypeOf = Object.setPrototypeOf ? Object.getPrototypeOf : function _getPrototypeOf(o) { return o.__proto__ || Object.getPrototypeOf(o); }; return _getPrototypeOf(o); }

/**
 * Web component for building page title
 */
var PageTitleElement = /*#__PURE__*/function (_HTMLElement) {
  _inherits(PageTitleElement, _HTMLElement);

  var _super = _createSuper(PageTitleElement);

  function PageTitleElement() {
    _classCallCheck(this, PageTitleElement);

    return _super.call(this);
  }
  /**
   * Call back function
   *
   * @return String html of page title to display
   */


  _createClass(PageTitleElement, [{
    key: "connectedCallback",
    value: function connectedCallback() {
      this._renderPageTitle();
    }
    /**
     * Retrieves HTML server logic built in jsp and
     * then combines with component styling
     *
     * @return String built html of page title to display
     */

  }, {
    key: "_renderPageTitle",
    value: function _renderPageTitle() {
      // no shadowdom, so slot info is being sent as innerHTML
      // contains html for page title.
      var pageTitleText = this.getAttribute('pageTitle');

      if (!pageTitleText) {
        pageTitleText = '';
      }

      var srOnly = this.getAttribute('data-sr-only');

      if (!srOnly) {
        srOnly = '';
      }

      var truncateTitle = this.hasAttribute('truncateTitle');
      var afterTitleHTML = this.innerHTML;

      var headerStyle = this._calculateHeaderStyle();

      var headingStyle = this._calculateHeadingStyle();

      var truncateStyle = this._calculateTruncateStyle(truncateTitle);

      this.innerHTML = "\n        <header class=\"".concat(headerStyle, "\">\n            <h1 class=\"").concat(headingStyle, " ").concat(srOnly, "\">\n                <span class=\"").concat(truncateStyle, "\">\n                    <span id=\"pageTitleHeader\">\n                        ").concat(pageTitleText, "\n                    </span>\n                </span>\n                ").concat(afterTitleHTML, "\n            </h1>\n            <style>\n                #authorDetailsPage {\n                    background: transparent !important;\n                }\n            </style>\n        </header>\n        ");
      return this.innerHTML;
    }
    /**
     * Function to determine truncate styling
     *
     * @return String
     */

  }, {
    key: "_calculateTruncateStyle",
    value: function _calculateTruncateStyle(truncateTitle) {
      return truncateTitle ? 'text-nowrap ellipsisOverflow truncateTitle' : '';
    }
    /**
     * Function to determine styling based upon subscription level
     *
     * @return String
     */

  }, {
    key: "_calculateHeaderStyle",
    value: function _calculateHeaderStyle() {
      return this._isGuestCheck() ? 'headerBackgroundPreview' : 'headerBackground';
    }
    /**
     * Function to determine styling based upon subscription level
     *
     * @return String
     */

  }, {
    key: "_calculateHeadingStyle",
    value: function _calculateHeadingStyle() {
      return this._isGuestCheck() ? 'documentHeader preview' : 'documentHeader';
    }
    /**
     * Function to determine user's subscribtion level
     *
     * @return {boolean} - true if guest, false if subscriber
     */

  }, {
    key: "_isGuestCheck",
    value: function _isGuestCheck() {
      var isGuest = true;

      if (scopus && scopus.platform && scopus.platform.user && scopus.platform.user.authorization) {
        isGuest = !scopus.platform.user.authorization.isSubscribed();
      }

      return isGuest;
    }
  }]);

  return PageTitleElement;
}( /*#__PURE__*/_wrapNativeSuper(HTMLElement));

customElements.define('sc-page-title', PageTitleElement);

/***/ })

/******/ });